"""Deploy the Copa NAO CCM MX 2026 penalty routine to real NAO V6 robots over Ethernet.

Used by the deploy dialog (deploy/deploy_dialog.py) and from the command line:

    python deploy/robot_deploy.py --list
    python deploy/robot_deploy.py --ping falcanao loki
    python deploy/robot_deploy.py --deploy falcanao:striker loki:keeper --team 49 \
        --striker-policy striker_v2 --keeper-policy vision_v2_balanced --start-mode buttonOnly
    python deploy/robot_deploy.py --status falcanao

Robots come from robot/Config/Robots/<name>/network.cfg (LAN address). Every remote
step runs in WSL (Windows) or bash (Linux) from the build tree of tools/robot_build.py
with robot/Make/Common/deployCopaPenalty. Wi-Fi is switched off by default (rule 2.7).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import robot_build  # noqa: E402

ROBOTS = ROOT / 'robot/Config/Robots'
POLICIES = ROOT / 'robot/Config/NeuralNets/CopaPenalty/policies.json'
ROLES = ('striker', 'keeper')
START_MODES = ('buttonOnly', 'whistleOnly', 'buttonOrWhistle')
CONFIGS = robot_build.CONFIGS
WORK_COMMON = robot_build.WORK + '/Make/Common'


@dataclass(frozen=True)
class Robot:
    name: str
    lan: str
    wlan: str


@dataclass
class DeployOptions:
    role: str = 'striker'
    striker_policy: str = 'striker_v2'
    keeper_policy: str = 'vision_v2_balanced'
    start_mode: str = 'buttonOnly'
    player: int = 1
    team: int = 0
    jumps: bool = True
    wifi_off: bool = True
    volume: int = 70
    restart: bool = True
    config: str = 'Release'
    dry_run: bool = False


def robots():
    """Robots of robot/Config/Robots/*/network.cfg, sorted by name."""
    found = []
    for network in sorted(ROBOTS.glob('*/network.cfg')):
        text = network.read_text(encoding='utf-8', errors='replace')
        values = dict(re.findall(r'^\s*(\w+)\s*=\s*"([^"]*)"\s*;', text, re.M))
        if values.get('lan'):
            found.append(Robot(values.get('name', network.parent.name), values['lan'], values.get('wlan', '')))
    return found


def policies():
    data = json.loads(POLICIES.read_text(encoding='utf-8'))
    return data


def validate(options):
    registry = policies()
    if options.role not in ROLES:
        raise ValueError('rol desconocido: %s' % options.role)
    if options.striker_policy not in registry['striker']:
        raise ValueError('politica de tirador desconocida: %s' % options.striker_policy)
    if options.keeper_policy not in registry['keeper']:
        raise ValueError('politica de portero desconocida: %s' % options.keeper_policy)
    if options.start_mode not in START_MODES:
        raise ValueError('modo de arranque desconocido: %s' % options.start_mode)
    if not 1 <= options.player <= 20:
        raise ValueError('numero de jugador fuera de 1..20')
    if options.config not in CONFIGS:
        raise ValueError('configuracion desconocida: %s' % options.config)


def deploy_command(robot, options):
    validate(options)
    args = ['./deployCopaPenalty', options.config, robot.lan, '-n', robot.name, '-p', str(options.player),
            '--role', options.role, '--striker-policy', options.striker_policy,
            '--keeper-policy', options.keeper_policy, '--start-mode', options.start_mode,
            '--jumps', 'on' if options.jumps else 'off', '-v', str(options.volume)]
    if options.team:
        args += ['-t', str(options.team)]
    if options.wifi_off:
        args += ['-w', 'NONE']
    if options.restart:
        args.append('-b')
    if options.dry_run:
        args.append('--dry-run')
    return 'cd "%s" && %s' % (WORK_COMMON, ' '.join(shlex.quote(a) for a in args))


def ssh_command(robot, remote):
    return ('cd "%s" && source ./setupSSH && ssh $SSHOPTIONS -o ConnectTimeout=4 nao@%s %s'
            % (WORK_COMMON, shlex.quote(robot.lan), shlex.quote(remote)))


def stream(script, on_line, distro=robot_build.DEFAULT_DISTRO):
    """Run a bash script (WSL on Windows) and pass every output line to on_line. Returns the exit code."""
    if sys.platform == 'win32':
        command = ['wsl.exe', '-d', distro, '--exec', 'bash', '-lc', script]
    else:
        command = ['bash', '-lc', script]
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding='utf-8', errors='replace', creationflags=flags)
    assert process.stdout is not None
    with process.stdout:
        for line in process.stdout:
            on_line(line.rstrip('\n'))
    return process.wait()


def ping(robot):
    """True if the robot answers on its LAN address (Windows ping, no WSL needed)."""
    if sys.platform == 'win32':
        command = ['ping', '-n', '1', '-w', '800', robot.lan]
    else:
        command = ['ping', '-c', '1', '-W', '1', robot.lan]
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors='replace', creationflags=flags)
    return result.returncode == 0 and 'TTL=' in result.stdout.upper()


def sync(on_line):
    return stream(robot_build.sync_script(), on_line)


def build(config, on_line):
    return stream(robot_build.build_script(config, os.cpu_count() or 4), on_line)


def deploy(robot, options, on_line):
    return stream(deploy_command(robot, options), on_line)


def status(robot, on_line):
    remote = ('echo "bhuman: $(systemctl --user is-active bhuman.service)"; '
              'cat /home/nao/Config/penaltystack_deploy.json 2>/dev/null || echo "sin manifiesto de PenaltyStack"')
    return stream(ssh_command(robot, remote), on_line)


def service(robot, action, on_line):
    if action not in ('start', 'stop', 'restart'):
        raise ValueError(action)
    return stream(ssh_command(robot, 'systemctl --user %s bhuman.service && echo "bhuman %s ok"' % (action, action)),
                  on_line)


def download_logs(robot, destination, on_line):
    destination = Path(destination).resolve() / robot.name
    destination.mkdir(parents=True, exist_ok=True)
    script = ('cd "%s" && source ./setupSSH && rsync -rzce "ssh $SSHOPTIONS" nao@%s:/home/nao/logging/ "%s/"'
              % (WORK_COMMON, shlex.quote(robot.lan), robot_build.to_wsl_path(destination)))
    return stream(script, on_line)


def _parse_targets(values, known):
    by_name = {r.name.lower(): r for r in known}
    targets = []
    for value in values:
        name, _, role = value.partition(':')
        robot = by_name.get(name.lower())
        if robot is None:
            if re.match(r'^\d+\.\d+\.\d+\.\d+$', name):
                robot = Robot(name, name, '')
            else:
                raise SystemExit('robot desconocido: %s' % name)
        targets.append((robot, role or None))
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description='Despliegue Copa NAO CCM MX 2026 (Tiro Penal) por Ethernet.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true', help='robots conocidos y politicas disponibles')
    action.add_argument('--ping', nargs='+', metavar='ROBOT')
    action.add_argument('--status', nargs='+', metavar='ROBOT')
    action.add_argument('--stop', nargs='+', metavar='ROBOT')
    action.add_argument('--start', nargs='+', metavar='ROBOT')
    action.add_argument('--deploy', nargs='+', metavar='ROBOT[:ROLE]')
    action.add_argument('--logs', nargs='+', metavar='ROBOT')
    parser.add_argument('--config', choices=CONFIGS, default='Release')
    parser.add_argument('--no-build', action='store_true', help='usar el binario ya compilado')
    parser.add_argument('--team', type=int, default=0)
    parser.add_argument('--first-player', type=int, default=1)
    parser.add_argument('--striker-policy', default=None)
    parser.add_argument('--keeper-policy', default=None)
    parser.add_argument('--start-mode', choices=START_MODES, default='buttonOnly')
    parser.add_argument('--no-jumps', action='store_true', help='portero sin estiradas laterales')
    parser.add_argument('--keep-wifi', action='store_true', help='no apagar el Wi-Fi (prohibido en competencia)')
    parser.add_argument('--volume', type=int, default=70)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--logs-dir', default=str(ROOT / 'runs/robot_logs'))
    args = parser.parse_args(argv)

    known = robots()
    registry = policies()
    lock = threading.Lock()

    def printer(prefix):
        def emit(line):
            with lock:
                print('[%s] %s' % (prefix, line), flush=True)
        return emit

    if args.list:
        for robot in known:
            print('%-12s LAN %-15s WLAN %s' % (robot.name, robot.lan, robot.wlan))
        for role in ROLES:
            for name, entry in registry[role].items():
                print('%-8s %-24s %-12s %s' % (role, name, entry['status'], entry['description']))
        return 0
    if args.ping:
        codes = []
        for robot, _ in _parse_targets(args.ping, known):
            ok = ping(robot)
            codes.append(ok)
            print('%-12s %-15s %s' % (robot.name, robot.lan, 'responde' if ok else 'SIN RESPUESTA'))
        return 0 if all(codes) else 1
    targets = _parse_targets(args.status or args.stop or args.start or args.deploy or args.logs, known)
    if args.status or args.stop or args.start or args.logs:
        def one(item):
            robot = item[0]
            if args.status:
                return status(robot, printer(robot.name))
            if args.logs:
                return download_logs(robot, args.logs_dir, printer(robot.name))
            return service(robot, 'stop' if args.stop else 'start', printer(robot.name))
        with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
            return max(pool.map(one, targets))

    if not args.no_build:
        if build(args.config, printer('build')) != 0:
            print('La compilacion fallo; no se despliega.', file=sys.stderr)
            return 1
    elif sync(printer('sync')) != 0:
        return 1
    jobs = []
    for index, (robot, role) in enumerate(targets):
        options = DeployOptions(role=role or 'striker',
                                striker_policy=args.striker_policy or registry['defaults']['striker'],
                                keeper_policy=args.keeper_policy or registry['defaults']['keeper'],
                                start_mode=args.start_mode, player=args.first_player + index, team=args.team,
                                jumps=not args.no_jumps, wifi_off=not args.keep_wifi, volume=args.volume,
                                config=args.config, dry_run=args.dry_run)
        validate(options)
        jobs.append((robot, options))
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        codes = list(pool.map(lambda job: deploy(job[0], job[1], printer(job[0].name)), jobs))
    for (robot, _), code in zip(jobs, codes):
        print('%-12s %s' % (robot.name, 'DESPLEGADO' if code == 0 else 'ERROR (codigo %d)' % code))
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == '__main__':
    sys.exit(main())
