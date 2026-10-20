"""Drive orange-ball calibration captures on the team's NAO v6 robots from the PC.

Uploads tools/nao_capture_ball.py, stops bhuman (it holds the camera through V4L2),
captures one labeled batch per camera and fetches the frames into
runs/ball_calibration/<robot>/<label>/. Threshold tuning then runs offline with
tools/tune_ball_thresholds.py; full workflow in docs/CALIBRACION_BALON.md.

    python tools/capture_ball_calibration.py --list
    python tools/capture_ball_calibration.py --push falcanao
    python tools/capture_ball_calibration.py --capture falcanao --label ball_130 --count 20
    python tools/capture_ball_calibration.py --capture falcanao --label bg_cancha
    python tools/capture_ball_calibration.py --fetch falcanao --label ball_130
    python tools/tune_ball_thresholds.py --captures runs/ball_calibration/falcanao

Remote access runs in WSL exactly like deploy/robot_deploy.py (setupSSH + cable, rule 2.7).
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
# OJO: no insertar ROOT/'deploy' en sys.path: deploy/deploy.py sombrearía el paquete
# de espacio de nombres `deploy` y rompería `from deploy.package import ...` en la suite.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from deploy import robot_deploy  # noqa: E402
import robot_build  # noqa: E402

REMOTE_ROOT = '/home/nao/ball_calibration'
LOCAL_ROOT = ROOT / 'runs/ball_calibration'
CAPTURE_SCRIPT = ROOT / 'tools/nao_capture_ball.py'
FORMATS = ('auto', 'yuv422', 'rgb')
RESOLUTIONS = (0, 1, 2, 3)


class CaptureOptions:
    """Un lote de capturas; los valores viajan al script del robot."""

    def __init__(self, label, cameras='0,1', count=20, interval_ms=200, warmup=15,
                 resolution=2, format='auto', manage_service=True):
        self.label = label
        self.cameras = cameras
        self.count = count
        self.interval_ms = interval_ms
        self.warmup = warmup
        self.resolution = resolution
        self.format = format
        self.manage_service = manage_service


def validate(options):
    if not re.match(r'^[A-Za-z0-9_-]{1,64}$', options.label or ''):
        raise ValueError('etiqueta invalida: %r (usa [A-Za-z0-9_-], p. ej. ball_130, bg_cancha)'
                         % options.label)
    if not re.match(r'^[01](,[01])?$', options.cameras):
        raise ValueError('camaras invalidas: %r (0=superior, 1=inferior)' % options.cameras)
    if not 1 <= options.count <= 200:
        raise ValueError('count fuera de 1..200')
    if not 50 <= options.interval_ms <= 5000:
        raise ValueError('interval-ms fuera de 50..5000')
    if not 0 <= options.warmup <= 100:
        raise ValueError('warmup fuera de 0..100')
    if options.resolution not in RESOLUTIONS:
        raise ValueError('resolution desconocida: %s' % options.resolution)
    if options.format not in FORMATS:
        raise ValueError('formato desconocido: %s' % options.format)


def push_command(robot):
    """Sube el script de captura al robot (rsync sobre el setupSSH del árbol de build)."""
    return ('cd "%s" && source ./setupSSH && ssh $SSHOPTIONS -o ConnectTimeout=4 nao@%s '
            '"mkdir -p %s" && rsync -e "ssh $SSHOPTIONS" "%s" nao@%s:%s/nao_capture_ball.py'
            % (robot_deploy.WORK_COMMON, robot.lan, REMOTE_ROOT,
               robot_build.to_wsl_path(CAPTURE_SCRIPT), robot.lan, REMOTE_ROOT))


def capture_command(robot, options):
    """Detiene bhuman, captura el lote y rearma el servicio (salvo --keep-service)."""
    validate(options)
    remote = []
    if options.manage_service:
        remote.append('systemctl --user stop bhuman.service 2>/dev/null; sleep 1')
    remote.append('python %s/nao_capture_ball.py --label %s --cameras %s --count %d '
                  '--interval-ms %d --warmup %d --resolution %d --format %s --out %s'
                  % (REMOTE_ROOT, options.label, options.cameras, options.count,
                     options.interval_ms, options.warmup, options.resolution, options.format,
                     REMOTE_ROOT))
    remote.append('rc=$?')
    if options.manage_service:
        remote.append('systemctl --user start bhuman.service')
    remote.append('exit $rc')
    return robot_deploy.ssh_command(robot, '; '.join(remote))


def fetch_command(robot, label, destination):
    """Trae un lote (o todos, label=None) a runs/ball_calibration/<robot>/<label>/.*/"""
    if label is not None and not re.match(r'^[A-Za-z0-9_-]{1,64}$', label):
        raise ValueError('etiqueta invalida: %r' % label)
    remote_path = '%s/%s/' % (REMOTE_ROOT, label) if label else '%s/' % REMOTE_ROOT
    return ('cd "%s" && source ./setupSSH && rsync -rzce "ssh $SSHOPTIONS" nao@%s:%s "%s/"'
            % (robot_deploy.WORK_COMMON, robot.lan, remote_path,
               robot_build.to_wsl_path(Path(destination))))


def local_destination(robot, label):
    return LOCAL_ROOT / robot.name / label if label else LOCAL_ROOT / robot.name


def resolve(names, known):
    by_name = {robot.name.lower(): robot for robot in known}
    resolved = []
    for name in names:
        robot = by_name.get(name.lower())
        if robot is None:
            raise SystemExit('robot desconocido: %s (python tools/capture_ball_calibration.py --list)' % name)
        resolved.append(robot)
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Capturas de calibracion del balon naranja en los NAO (ver docs/CALIBRACION_BALON.md).')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true', help='robots conocidos')
    action.add_argument('--push', nargs='+', metavar='ROBOT', help='subir el script de captura')
    action.add_argument('--capture', nargs='+', metavar='ROBOT', help='capturar un lote')
    action.add_argument('--fetch', nargs='+', metavar='ROBOT', help='traer las capturas al PC')
    parser.add_argument('--label', default=None, help='lote: ball_020, ball_130, ball_200, bg_cancha…')
    parser.add_argument('--cameras', default='0,1', help='0=superior, 1=inferior (omision: ambas)')
    parser.add_argument('--count', type=int, default=20, help='frames por camara')
    parser.add_argument('--interval-ms', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--resolution', type=int, default=2, help='2=VGA 640x480 (runtime B-Human)')
    parser.add_argument('--format', choices=FORMATS, default='auto')
    parser.add_argument('--keep-service', action='store_true',
                        help='no detener/reiniciar bhuman (solo si ya esta detenido)')
    args = parser.parse_args(argv)

    known = robot_deploy.robots()
    if args.list:
        for robot in known:
            print('%-12s LAN %-15s WLAN %s' % (robot.name, robot.lan, robot.wlan))
        return 0

    if args.capture and not args.label:
        parser.error('--capture necesita --label')
    options = CaptureOptions(label=args.label, cameras=args.cameras, count=args.count,
                             interval_ms=args.interval_ms, warmup=args.warmup,
                             resolution=args.resolution, format=args.format,
                             manage_service=not args.keep_service)
    if args.capture:
        validate(options)

    targets = resolve(args.push or args.capture or args.fetch, known)
    lock = threading.Lock()

    def printer(prefix):
        def emit(line):
            with lock:
                print('[%s] %s' % (prefix, line), flush=True)
        return emit

    def one(robot):
        on_line = printer(robot.name)
        if args.push:
            return robot_deploy.stream(push_command(robot), on_line)
        if args.capture:
            if robot_deploy.stream(push_command(robot), on_line) != 0:
                return 1
            return robot_deploy.stream(capture_command(robot, options), on_line)
        destination = local_destination(robot, args.label)
        destination.mkdir(parents=True, exist_ok=True)
        return robot_deploy.stream(fetch_command(robot, args.label, destination), on_line)

    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        codes = list(pool.map(one, targets))
    for robot, code in zip(targets, codes):
        print('%-12s %s' % (robot.name, 'OK' if code == 0 else 'FALLO (%d)' % code))
    return max(codes) if codes else 0


if __name__ == '__main__':
    sys.exit(main())
