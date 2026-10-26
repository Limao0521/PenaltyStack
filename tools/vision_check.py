"""Pruebas de visión del balón Trionda para tirador y portero en MuJoCo.

Renderiza las cámaras de cabeza simuladas (CameraBottom, QVGA 320x240 como el
pipeline onboard de penalty/nao.py) y ejecuta penalty.vision.detect_orange_ball
contra el suelo verdadero de la física. La FSM del ensayo sigue usando
percepción ideal; esta herramienta mide sólo el detector naranja frente a la
apariencia Trionda simulada (balón de 65 mm, naranja con manchas oscuras).

Uso:
    .venv/bin/python tools/vision_check.py            # escenarios estáticos
    .venv/bin/python tools/vision_check.py --trial    # + penal completo en vivo
    .venv/bin/python tools/vision_check.py --out runs/vision

Escribe un informe JSON y capturas PNG anotadas bajo --out (runs/vision).
Criterio de acierto (baseline sin calibrar, tolerancias explícitas):
    balón proyectado dentro del cuadro, detección presente,
    |rango - profundidad| <= max(0.05 m, 40%), |rumbo| <= 0.15 rad.
"""
import argparse
import json
import math
import os
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from penalty.motion_contract import ROOT
from penalty.physical_sim import PhysicalSimulator
from penalty.vision import detect_orange_ball

WIDTH, HEIGHT = 320, 240          # QVGA, como el ALVideoDevice onboard
FOCAL = (WIDTH / 2.0) / math.tan(math.radians(54.7) / 2.0)
# El detector onboard usa stride=2 por CPU; aquí se comprueba a resolución
# completa y con tolerancias de baseline sin calibrar (vision.py lo avisa).
STRIDE = 1
RANGE_RTOL, RANGE_ATOL, BEARING_TOL = 0.40, 0.05, 0.15
CAMERA = 'CameraBottom'

# Poses de cabeza explícitas: la FSM onboard no cabecea hacia abajo (head_pitch
# queda en 0), así que cada escenario documenta la pose con la que se prueba.
# HeadPitch positivo = mirar abajo (eje +Y del modelo, igual que el NAO).
SCENARIOS = (
    # Tirador (nace en x=0.3 mirando hacia -x; el balón está a 0.2 m).
    {'role': 'striker', 'ball': (0.10, 0.00), 'head_yaw': 0.0, 'head_pitch': 0.45,
     'label': 'striker: balón en punto penal (0.20 m), cabeza abajo'},
    {'role': 'striker', 'ball': (0.10, 0.15), 'head_yaw': -0.64, 'head_pitch': 0.40,
     'label': 'striker: balón a la derecha (0.25 m)'},
    {'role': 'striker', 'ball': (-0.20, 0.00), 'head_yaw': 0.0, 'head_pitch': 0.10,
     'label': 'striker: balón a 0.50 m al frente'},
    {'role': 'striker', 'ball': (-0.50, -0.20), 'head_yaw': 0.245, 'head_pitch': 0.0,
     'label': 'striker: balón a 0.82 m adelante-izquierda'},
    # Portero (nace en x=-1.08 mirando hacia +x; el balón está a 1.18 m).
    {'role': 'keeper', 'ball': (0.10, 0.00), 'head_yaw': 0.0, 'head_pitch': 0.0,
     'label': 'keeper: balón en punto penal (1.18 m)'},
    {'role': 'keeper', 'ball': (0.10, 0.40), 'head_yaw': 0.33, 'head_pitch': 0.0,
     'label': 'keeper: balón a la izquierda (1.25 m)'},
    {'role': 'keeper', 'ball': (0.10, -0.40), 'head_yaw': -0.33, 'head_pitch': 0.0,
     'label': 'keeper: balón a la derecha (1.25 m)'},
    {'role': 'keeper', 'ball': (0.60, 0.00), 'head_yaw': 0.0, 'head_pitch': 0.0,
     'label': 'keeper: balón a 1.68 m (límite/fuera del FOV inferior)'},
)


def create_renderer(model):
    """Renderer offscreen; prueba backends GL sin display en orden."""
    import mujoco
    candidates = ([os.environ['MUJOCO_GL']] if os.environ.get('MUJOCO_GL')
                  else ['egl', 'osmesa', 'glfw'])
    failures = []
    for backend in candidates:
        os.environ['MUJOCO_GL'] = backend
        try:
            return mujoco.Renderer(model, HEIGHT, WIDTH), backend
        except Exception as error:  # noqa: BLE001 - el siguiente backend decide
            failures.append('%s: %s' % (backend, error))
    raise RuntimeError('Sin backend GL offscreen disponible (%s). '
                       'Instala libEGL/libOSMesa o ejecuta con display.'
                       % '; '.join(failures))


def write_png(path, rgb):
    """PNG RGB8 mínimo sin dependencias externas."""
    height, width = rgb.shape[:2]
    raw = b''.join(b'\x00' + bytes(rgb[y]) for y in range(height))

    def chunk(tag, payload):
        return (struct.pack('>I', len(payload)) + tag + payload +
                struct.pack('>I', zlib.crc32(tag + payload) & 0xffffffff))

    png = (b'\x89PNG\r\n\x1a\n' +
           chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
           chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
    Path(path).write_bytes(png)


def orange_bbox(rgb, stride=STRIDE):
    """Caja del mayor grupo naranja (mismo umbral que vision.py), para anotar."""
    import numpy as np
    view = rgb[::stride, ::stride].astype(np.int16)
    r, g, b = view[:, :, 0], view[:, :, 1], view[:, :, 2]
    mask = ((r >= 115) & (g >= 45) & (g <= 200) & (b <= 120) &
            (r > g * 1.18) & (g > b * 1.15))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min() * stride), int(ys.min() * stride),
            int(xs.max() * stride) + stride, int(ys.max() * stride) + stride)


def annotate(rgb, bbox):
    import numpy as np
    if bbox is None:
        return rgb
    out = rgb.copy()
    x0, y0, x1, y1 = bbox
    for x in range(x0, min(x1, WIDTH)):
        for y in (y0, min(y1 - 1, HEIGHT - 1)):
            out[y, x] = (255, 255, 255)
    for y in range(y0, min(y1, HEIGHT)):
        for x in (x0, min(x1 - 1, WIDTH - 1)):
            out[y, x] = (255, 255, 255)
    return out


def camera_truth(sim, role, camera=CAMERA):
    """Suelo verdadero en marco de cámara: profundidad, rumbo y pixel esperado."""
    import numpy as np
    cam = sim.model.camera(role + '_' + camera)
    position = sim.data.cam_xpos[cam.id]
    rotation = sim.data.cam_xmat[cam.id].reshape(3, 3)
    ball = sim.data.qpos[sim.ball_qpos:sim.ball_qpos + 3]
    local = rotation.T @ (ball - position)
    depth = float(-local[2])            # adelante de la cámara (+X cam mira -Z)
    bearing = -math.atan2(float(local[0]), depth) if depth > 1e-6 else None
    pixel = (WIDTH / 2.0 + FOCAL * float(local[0]) / depth,
             HEIGHT / 2.0 - FOCAL * float(local[1]) / depth) if depth > 1e-6 else None
    radius_px = FOCAL * 0.0325 / depth if depth > 1e-6 else 0.
    # Tres categorías: disco completo dentro (se evalúa), disco recortado por
    # el borde (detección esperada pero rango/rumbo no fiables) y fuera.
    in_frame = bool(pixel and radius_px <= pixel[0] < WIDTH - radius_px and
                    radius_px <= pixel[1] < HEIGHT - radius_px and depth > 0)
    intersects = bool(pixel and -radius_px < pixel[0] < WIDTH + radius_px and
                      -radius_px < pixel[1] < HEIGHT + radius_px and depth > 0)
    return {'depth_m': depth, 'bearing_rad': bearing, 'pixel': pixel,
            'radius_px': radius_px, 'in_frame': in_frame, 'intersects': intersects,
            'slant_m': float(np.linalg.norm(ball - position))}


def set_head(sim, role, yaw, pitch):
    import mujoco
    for joint, value in ((role + '_HeadYaw', yaw), (role + '_HeadPitch', pitch)):
        address = sim.model.joint(joint).qposadr[0]
        low, high = sim.model.joint(joint).range
        sim.data.qpos[address] = max(low, min(high, value))


def place_ball(sim, xy):
    sim.data.qpos[sim.ball_qpos:sim.ball_qpos + 7] = [xy[0], xy[1], 0.0325, 1, 0, 0, 0]


def evaluate(sim, renderer, role, out_dir, tag):
    import mujoco
    cam_id = sim.model.camera(role + '_' + CAMERA).id
    with sim.mutation_lock():
        mujoco.mj_forward(sim.model, sim.data)
        renderer.update_scene(sim.data, cam_id)
        pixels = renderer.render()
    truth = camera_truth(sim, role)
    detection = detect_orange_ball(pixels.tobytes(), WIDTH, HEIGHT, stride=STRIDE)
    row = {'role': role, 'camera': role + '_' + CAMERA,
           'detected': detection is not None, 'truth': truth}
    if detection:
        row['detection'] = {'range_m': detection[0], 'bearing_rad': detection[1],
                            'pixels': detection[2]}
    if truth['in_frame']:
        row['range_error_m'] = (abs(detection[0] - truth['depth_m'])
                                if detection else None)
        row['bearing_error_rad'] = (abs(detection[1] - truth['bearing_rad'])
                                    if detection else None)
        row['pass'] = bool(detection and
                           row['range_error_m'] <= max(RANGE_ATOL, RANGE_RTOL * truth['depth_m']) and
                           row['bearing_error_rad'] <= BEARING_TOL)
    elif truth['intersects']:
        row['pass'] = None  # disco recortado por el borde: fuera de estadística
    else:
        row['pass'] = not detection  # fuera del cuadro: no debe haber detección
    if out_dir:
        name = out_dir / ('%s_%s.png' % (tag, role))
        write_png(name, annotate(pixels, orange_bbox(pixels)))
        row['image'] = str(name)
    return row


def run_static(out_dir):
    sim = PhysicalSimulator(seed=0)
    renderer, backend = create_renderer(sim.model)
    rows = []
    try:
        for index, scenario in enumerate(SCENARIOS):
            sim.reset_world()
            place_ball(sim, scenario['ball'])
            set_head(sim, scenario['role'], scenario['head_yaw'], scenario['head_pitch'])
            row = evaluate(sim, renderer, scenario['role'], out_dir,
                           'static%02d' % index)
            row['scenario'] = scenario['label']
            row['ball_xy'] = list(scenario['ball'])
            row['head_yaw'] = scenario['head_yaw']
            row['head_pitch'] = scenario['head_pitch']
            rows.append(row)
    finally:
        renderer.close()
        sim.close()
    return rows, backend


def run_trial(out_dir, seconds, every):
    """Penal completo con el motor original; muestrea ambas cámaras."""
    from penalty.sim_session import SimulationSession
    sim = PhysicalSimulator(seed=0)
    session = SimulationSession(sim, record_dir=None, mode='penalty')
    renderer, backend = create_renderer(sim.model)
    samples = []
    try:
        session.command('start')
        next_sample = every
        budget = int((seconds + 8) / sim.dt) + 1
        for _ in range(budget):
            session.tick()
            if session.state in ('FINISHED', 'ABORTED', 'ERROR', 'CLOSED'):
                break
            if session.state == 'RUNNING' and session.penalty_time >= seconds - 1e-9:
                session._finish()
                session._publish()
                break
            if session.state != 'RUNNING' or session.penalty_time < next_sample:
                continue
            next_sample += every
            for role in ('striker', 'keeper'):
                row = evaluate(sim, renderer, role, None, '')
                row['t_s'] = round(session.penalty_time, 3)
                output = sim.outputs.get(role, {})
                row['motion'] = output.get('motion', 'stand')
                row['phase'] = output.get('phase', 'not_started')
                samples.append(row)
            if out_dir and (len(samples) // 2) in (1, 4, 8):
                for role in ('striker', 'keeper'):
                    evaluate(sim, renderer, role, out_dir,
                             'trial_%04.1fs' % session.penalty_time)
        final = session.snapshot()
    finally:
        session.close()
        session.tick()
        renderer.close()
        sim.close()
    return samples, final, backend


def summarize(rows):
    summary = {}
    for role in ('striker', 'keeper'):
        own = [r for r in rows if r['role'] == role]
        visible = [r for r in own if r['truth']['in_frame']]
        clipped = [r for r in own if not r['truth']['in_frame'] and r['truth']['intersects']]
        passed = [r for r in own if r.get('pass') is True]
        failed = [r for r in own if r.get('pass') is False]
        range_errors = [r['range_error_m'] for r in visible if r.get('range_error_m') is not None]
        bearing_errors = [r['bearing_error_rad'] for r in visible if r.get('bearing_error_rad') is not None]
        summary[role] = {'muestras': len(own), 'balón_en_cuadro': len(visible),
                         'recortadas_borde': len(clipped), 'fallos': len(failed),
                         'detecciones_correctas': len(passed),
                         'error_rango_med_m': (sum(range_errors) / len(range_errors)) if range_errors else None,
                         'error_rumbo_med_rad': (sum(bearing_errors) / len(bearing_errors)) if bearing_errors else None}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=str, default=str(ROOT / 'runs' / 'vision'))
    parser.add_argument('--trial', action='store_true',
                        help='ejecuta además un penal completo muestreando las cámaras')
    parser.add_argument('--seconds', type=float, default=28.)
    parser.add_argument('--every', type=float, default=0.5,
                        help='segundos entre muestras del penal')
    args = parser.parse_args()
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    static_rows, backend = run_static(out_dir)
    report = {'backend_gl': backend, 'camara': CAMERA, 'resolucion': [WIDTH, HEIGHT],
              'criterio': {'rango': 'max(%.2f m, %.0f%%)' % (RANGE_ATOL, RANGE_RTOL * 100),
                          'rumbo_rad': BEARING_TOL},
              'statico': static_rows, 'statico_resumen': summarize(static_rows)}
    print('== Escenarios estáticos (balón Trionda simulado, %s) ==' % backend)
    for row in static_rows:
        truth = row['truth']
        detection = row.get('detection')
        print('%-58s cuadro=%-5s detección=%-22s %s' % (
            row['scenario'], truth['in_frame'],
            ('%.2f m / %+.2f rad / %d px' % (detection['range_m'],
                                             detection['bearing_rad'], detection['pixels']))
            if detection else 'ninguna',
            'OK' if row['pass'] else ('BORDE' if row['pass'] is None else 'FALLO')))

    if args.trial:
        samples, final, backend = run_trial(out_dir, args.seconds, args.every)
        report['penal'] = {'muestras': samples, 'resumen': summarize(samples),
                           'resultado_final': final.get('result'),
                           'portero': {k: final['robots']['keeper'].get(k)
                                       for k in ('motion', 'phase', 'fallen')}}
        print('== Penal en vivo (cada %.1f s) ==' % args.every)
        summary = report['penal']['resumen']
        for role, values in summary.items():
            print('%-8s ' % role +
                  'muestras=%(muestras)d en_cuadro=%(balón_en_cuadro)d '
                  'recortadas=%(recortadas_borde)d correctas=%(detecciones_correctas)d '
                  'fallos=%(fallos)d err_rango_med=%(error_rango_med_m)s '
                  'err_rumbo_med=%(error_rumbo_med_rad)s' % values)
        print('Resultado del penal: %s · portero: %s/%s caído=%s' % (
            final.get('result'), final['robots']['keeper'].get('motion'),
            final['robots']['keeper'].get('phase'), final['robots']['keeper'].get('fallen')))

    if out_dir:
        report_path = out_dir / 'report.json'
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n',
                               encoding='utf-8')
        print('Informe: %s · capturas PNG en %s' % (report_path, out_dir))
    static_ok = all(row.get('pass') is not False for row in static_rows)
    return 0 if static_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
