"""Record the keeper's genuflectStandDefender in the B-Human MuJoCo simulation.

The keeper stands, blocks with the RoboCup genuflect to the left, stands up,
then repeats to the right. Writes an MP4 with a front and a side view and a
JSON summary (torso tilt, torso height, fall flag, executed keyframes).

    .venv/bin/python tools/record_keeper_block.py --output runs/keeper_block.mp4
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WIDTH, HEIGHT, FPS = 480, 360, 30

# (time from start in s, manual request): stand, block left, stand, block right, stand.
SCRIPT = [
    (0.0, {'motion': 'stand'}),
    (2.0, {'motion': 'block', 'block_side': 'defender_left'}),
    (5.0, {'motion': 'stand'}),
    (9.0, {'motion': 'block', 'block_side': 'defender_right'}),
    (12.0, {'motion': 'stand'}),
]
DURATION = 16.0


def renderer_for(model):
    import mujoco
    for backend in ([os.environ['MUJOCO_GL']] if os.environ.get('MUJOCO_GL') else ['egl', 'osmesa', 'glfw']):
        os.environ['MUJOCO_GL'] = backend
        try:
            return mujoco.Renderer(model, HEIGHT, WIDTH)
        except Exception:  # noqa: BLE001 - try the next GL backend
            continue
    raise RuntimeError('No offscreen GL backend (EGL/OSMesa) available')


def torso_tilt(data, body):
    """Pitch and roll of the torso in degrees, from its world rotation."""
    rotation = data.xmat[body].reshape(3, 3)
    up = rotation[:, 2]
    pitch = math.degrees(math.atan2(up[0], up[2]))
    roll = math.degrees(math.atan2(-up[1], up[2]))
    return pitch, roll


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default=str(ROOT / 'runs/keeper_block.mp4'))
    args = parser.parse_args()

    import cv2
    import mujoco
    from penalty.physical_sim import PhysicalSimulator
    from penalty.sim_session import SimulationSession

    sim = PhysicalSimulator(seed=0, ball_xy=(0.1, 0.0))
    session = SimulationSession(sim, record_dir=None, mode='manual')
    session.command('role', 'keeper')
    session.command('manual', SCRIPT[0][1])
    session.command('start')

    renderer = renderer_for(sim.model)
    body = sim.model.body('keeper_Nao').id
    views = []
    for azimuth in (180.0, 90.0):  # front (from the striker side), side
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.elevation, camera.azimuth = 1.3, -12.0, azimuth
        views.append(camera)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (2 * WIDTH, HEIGHT))
    summary = {'segments': [], 'max_abs_pitch_deg': 0.0, 'max_abs_roll_deg': 0.0,
               'min_torso_height_m': 9.0, 'fallen': False, 'keyframes': []}
    step, next_frame, started = 0, 0.0, None
    try:
        while True:
            session.tick()
            if session.state in ('ERROR', 'FINISHED', 'ABORTED', 'CLOSED'):
                raise RuntimeError(f'session ended early: {session.state} {session.error}')
            if session.state != 'RUNNING':
                continue
            now = float(sim.data.time)
            if started is None:
                started = now
            t = now - started
            while step + 1 < len(SCRIPT) and t >= SCRIPT[step + 1][0]:
                step += 1
                session.command('manual', SCRIPT[step][1])
            if t >= DURATION:
                break

            pitch, roll = torso_tilt(sim.data, body)
            height = float(sim.data.xpos[body][2])
            keeper = session.sim.outputs.get('keeper', {})
            keyframe = keeper.get('motion_info', {}).get('keyframe', '')
            fallen = bool(sim.snapshot()['robots']['keeper'].get('fallen'))
            summary['max_abs_pitch_deg'] = max(summary['max_abs_pitch_deg'], abs(pitch))
            summary['max_abs_roll_deg'] = max(summary['max_abs_roll_deg'], abs(roll))
            summary['min_torso_height_m'] = min(summary['min_torso_height_m'], height)
            summary['fallen'] |= fallen
            phase = keeper.get('phase', '')
            if phase == 'keyframeMotion' and keyframe and (not summary['keyframes'] or summary['keyframes'][-1] != keyframe):
                summary['keyframes'].append(keyframe)

            if t >= next_frame:
                next_frame += 1.0 / FPS
                panels = []
                for camera in views:
                    camera.lookat[:] = sim.data.xpos[body] - np.array([0.0, 0.0, 0.12])
                    renderer.update_scene(sim.data, camera)
                    panels.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
                frame = np.hstack(panels)
                request = SCRIPT[step][1]
                label = request['motion'] + (' ' + request['block_side'] if 'block_side' in request else '')
                lines = [f't={t:4.1f}s  request: {label}',
                         f'phase: {phase}  keyframe: {keyframe if phase == "keyframeMotion" else "-"}',
                         f'torso pitch {pitch:+5.1f}  roll {roll:+5.1f} deg  height {height:.2f} m'
                         + ('  FALLEN' if fallen else '')]
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'front', (8, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, 'side', (WIDTH + 8, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                writer.write(frame)
                if abs(t - 3.5) < 0.5 / FPS or abs(t - 10.5) < 0.5 / FPS:
                    cv2.imwrite(str(output.with_name(output.stem + f'_{t:.1f}s.png')), frame)
    finally:
        writer.release()
        session.close()
        session.tick()

    for key in ('max_abs_pitch_deg', 'max_abs_roll_deg', 'min_torso_height_m'):
        summary[key] = round(summary[key], 3)
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
