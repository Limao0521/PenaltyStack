"""Validate and show original B-Human keeper defenses in free MuJoCo physics.

No hardware, ball kick, learned joint trajectories or effectiveness estimate.
Each case starts standing, uses the public manual request, and then asks the
original engine to recover. Rendering hides the striker without changing its
physics. Sources, motion parameters and published policies remain untouched.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from penalty.keeper_defenses import (DEFENSE_CATALOG_VERSION, DEFENSE_MOTIONS,
                                     DEFENSE_NAMES, PREPARATION_NAME, catalog)
from penalty.physical_sim import PhysicalSimulator
from penalty.sim_session import SimulationSession

VERSION = 'keeper_defenses_physical_validation_v1'
LABELS = {'center': 'Central izquierda', 'center_right': 'Central derecha',
          'wide_left': 'Amplia izquierda', 'wide_right': 'Amplia derecha',
          'left': 'Salto izquierdo', 'right': 'Salto derecho',
          'prepare': 'Preparación: sentarse'}
WIDTH, HEIGHT = 600, 400
CAPTURE_TIMES = (0., .36, .72, 1.2, 2.4)


def _font(size, bold=False):
    for path in (Path('C:/Windows/Fonts') / ('arialbd.ttf' if bold else 'arial.ttf'),
                 Path('/usr/share/fonts/truetype/dejavu') /
                 ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _frame_record(sim, began):
    output = sim.outputs['keeper']
    request = sim.last_inputs.get('keeper', {}).get('request', {})
    info = output.get('motion_info', {})
    return {'time_s': round(float(sim.data.time) - began, 6),
            'request_motion': request.get('motion'),
            'request_defense': request.get('block_side'),
            'phase': output.get('phase'), 'keyframe': info.get('keyframe'),
            'mirror': info.get('mirror'),
            'fall_state': output.get('sensor_info', {}).get('fall_state'),
            'torso_position_m': sim.pose('keeper')[0].tolist(),
            'stable': bool(sim.stable())}


def validate_case(name, output, seed=71001, hold_s=3., render=True):
    """Return a validation summary and presentation frames; always reap children."""
    if name not in DEFENSE_MOTIONS:
        raise ValueError('unknown defense')
    if not math.isfinite(hold_s) or not 2.4 <= hold_s <= 3.6:
        raise ValueError('hold_s must be 2.4..3.6 seconds')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sim = PhysicalSimulator(seed=seed, ball_xy=(1., .8))
    session = SimulationSession(sim, record_dir=None, mode='manual', publish_ticks=False)
    processes, frames, animation, records = [], [], [], []
    renderer = None
    summary = {'name': name, 'kind': 'preparation' if name == PREPARATION_NAME else 'defense',
               'expected_request': DEFENSE_MOTIONS[name][0],
               'expected_keyframe': DEFENSE_MOTIONS[name][1],
               'expected_mirror': DEFENSE_MOTIONS[name][2], 'seed': seed,
               'hold_s': hold_s, 'executed_on_hardware': False}
    try:
        session.command('role', 'keeper')
        session.command('start')
        processes = [bridge.process for bridge in sim.bridges.values()]
        hello = sim.bridges['keeper'].hello
        summary['runtime_capabilities'] = hello.get('capabilities', [])
        summary['source_commit'] = hello.get('source_commit')
        if DEFENSE_CATALOG_VERSION not in summary['runtime_capabilities']:
            raise RuntimeError('rebuild motion runtime: missing keeper_defenses_v2')
        declared = hello.get('keeper_preparations' if name == PREPARATION_NAME else 'keeper_defenses', [])
        if name not in declared:
            raise RuntimeError('native runtime did not advertise the request')
        for role in ('striker', 'keeper'):
            root = sim.model.joint(role + '_root_free')
            if root.type[0] != mujoco.mjtJoint.mjJNT_FREE:
                raise RuntimeError('validation requires free robot roots')
        for _ in range(418):
            session.tick()
            if session.state == 'RUNNING':
                break
        else:
            raise RuntimeError('both robots failed to settle within five seconds')
        summary['settle_time_s'] = float(sim.data.time)
        began = float(sim.data.time)
        joints = [joint for joint in sim.adapter.joints['keeper'] if joint is not None]
        joint_addresses = [joint.qposadr[0] for joint in joints]
        initial_angles = sim.data.qpos[joint_addresses].copy()
        max_joint_change = 0.
        if render:
            for geom in range(sim.model.ngeom):
                body_name = sim.model.body(sim.model.geom_bodyid[geom]).name
                if body_name.startswith('striker_'):
                    sim.model.geom_group[geom] = 5
            option = mujoco.MjvOption()
            option.geomgroup[5] = 0
            renderer = mujoco.Renderer(sim.model, height=HEIGHT, width=WIDTH)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [-1.10, 0., .30]
            camera.distance, camera.azimuth, camera.elevation = 1.55, 0., -8.
        session.command('manual', {'motion': 'block', 'block_side': name})
        next_capture, native_matches, active_frames = 0, 0, 0
        mirror_transients = []
        last_active_mirror = None
        hold_ticks = math.ceil(hold_s / sim.dt)
        for tick in range(hold_ticks + 1):
            row = _frame_record(sim, began)
            records.append(row)
            max_joint_change = max(max_joint_change, float(np.max(np.abs(
                sim.data.qpos[joint_addresses] - initial_angles))))
            expected = row['phase'] == 'keyframeMotion' and row['keyframe'] == summary['expected_keyframe']
            if expected:
                active_frames += 1
                last_active_mirror = row['mirror']
                if row['mirror'] == summary['expected_mirror']:
                    native_matches += 1
                else:
                    # Upstream reports the new keyframe before its initial
                    # update sets isMirror. Keep that transient observable;
                    # demand matching accepted frames and the final active
                    # frame instead of declaring the first output an error.
                    mirror_transients.append({'time_s': row['time_s'], 'mirror': row['mirror']})
            capture = next_capture < len(CAPTURE_TIMES) and row['time_s'] + 1e-9 >= CAPTURE_TIMES[next_capture]
            if renderer is not None and (tick % 4 == 0 or capture):
                renderer.update_scene(sim.data, camera=camera, scene_option=option)
                image = Image.fromarray(renderer.render().copy())
                if tick % 4 == 0:
                    animation.append((row['time_s'], image.copy(), row['phase']))
                while next_capture < len(CAPTURE_TIMES) and row['time_s'] + 1e-9 >= CAPTURE_TIMES[next_capture]:
                    frames.append((row['time_s'], image.copy(), row['phase']))
                    next_capture += 1
            if tick < hold_ticks:
                session.tick()
        summary['matching_keyframe_frames'] = native_matches
        summary['keyframe_frames'] = active_frames
        summary['mirror_transients'] = mirror_transients
        summary['last_active_mirror'] = last_active_mirror
        summary['max_joint_change_rad'] = max_joint_change
        summary['motion_executed'] = (native_matches > 0 and max_joint_change > .1
                                      and last_active_mirror == summary['expected_mirror'])
        summary['manual_recovery_request_time_s'] = round(float(sim.data.time) - began, 6)
        summary['release_behavior'] = ('automatic getup after original jump is accepted'
                                       if name in ('left', 'right') else 'hold guard then manual getup')
        session.command('manual', {'motion': 'getup'})
        stable_for, recovered_at, stable_hold_s = 0., None, 0.
        interrupted_attempts = []
        while session.penalty_time < 28. - 1e-9:
            session.tick()
            row = _frame_record(sim, began)
            records.append(row)
            upright = (row['fall_state'] == 'upright' and row['phase'] == 'stand'
                       and row['stable'] and row['torso_position_m'][2] > .28)
            if recovered_at is None:
                stable_for = stable_for + sim.dt if upright else 0.
                if stable_for >= .5:
                    recovered_at = row['time_s']
                    session.command('manual', {'motion': 'stand'})
                    if renderer is not None:
                        renderer.update_scene(sim.data, camera=camera, scene_option=option)
                        Image.fromarray(renderer.render().copy()).save(output / (name + '_recovered.png'))
            else:
                frame = sim.adapter.frame('keeper')
                remained_upright = (row['fall_state'] == 'upright' and row['phase'] == 'stand'
                                    and sim.pose('keeper')[0][2] > .28
                                    and sum(frame['fsr'][:4]) > .05 and sum(frame['fsr'][4:]) > .05)
                if not remained_upright:
                    interrupted_attempts.append({'gate_time_s': recovered_at, 'lost_time_s': row['time_s']})
                    stable_for, recovered_at, stable_hold_s = 0., None, 0.
                else:
                    stable_hold_s += sim.dt
                    if stable_hold_s >= 1.:
                        break
        summary['recovered_to_stable_stand'] = recovered_at is not None and stable_hold_s >= 1.
        summary['stable_stand_time_s'] = recovered_at
        summary['upright_confirmation_s'] = round(stable_hold_s, 6)
        summary['interrupted_recovery_attempts'] = interrupted_attempts
        summary['end_time_s'] = round(float(sim.data.time) - began, 6)
        summary['final_phase'] = sim.outputs['keeper'].get('phase')
        summary['final_fall_state'] = sim.outputs['keeper'].get('sensor_info', {}).get('fall_state')
        summary['phases_first_seen_s'] = {}
        for row in records:
            summary['phases_first_seen_s'].setdefault(row['phase'], row['time_s'])
        summary['passed'] = summary['motion_executed'] and summary['recovered_to_stable_stand']
    except Exception as error:
        summary['passed'] = False
        summary['error'] = str(error)
    finally:
        try:
            if renderer is not None:
                renderer.close()
        finally:
            try:
                sim.close()
            finally:
                summary['children_collected'] = all(process.poll() is not None for process in processes)
    trace = output / (name + '.frames.jsonl')
    trace.write_bytes(('\n'.join(json.dumps(row, allow_nan=False) for row in records) + '\n').encode('utf-8'))
    summary['trace'] = trace.name
    summary['trace_sha256'] = hashlib.sha256(trace.read_bytes()).hexdigest()
    summary['passed'] = summary['passed'] and summary['children_collected']
    return summary, frames, animation


def _gallery(rows, output, title):
    margin, top, row_header, frame_header = 24, 120, 54, 36
    row_height = row_header + frame_header + HEIGHT + margin
    sheet = Image.new('RGB', (5 * WIDTH + 2 * margin, top + len(rows) * row_height + 66), '#f5f7fa')
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 20), title, font=_font(37, True), fill='#17263a')
    draw.text((margin, 72), 'MuJoCo + B-Human original | Primitivas fijas; sin movimientos entrenados', font=_font(27), fill='#334155')
    for row_index, (name, frames) in enumerate(rows):
        y = top + row_index * row_height
        primitive, mirror = DEFENSE_MOTIONS[name][1:]
        label = LABELS[name] + ': ' + primitive + (' (espejado)' if mirror else '')
        draw.text((margin, y + 4), label, font=_font(29, True), fill='#17263a')
        for column, (elapsed, image, phase) in enumerate(frames):
            x = margin + column * WIDTH
            draw.text((x + 10, y + row_header), 't = %.2f s | %s' % (elapsed, phase), font=_font(23), fill='#334155')
            sheet.paste(image, (x, y + row_header + frame_header))
    draw.text((margin, sheet.height - 53), 'Vista desde el campo; el tirador se oculta solo en la imagen. Sin tiro de balón: no mide eficacia.', font=_font(26), fill='#334155')
    sheet.save(output)


def _animation(rows, output, title, columns=3):
    if not rows or any(not video for _, video in rows):
        return
    width, height, heading, footer = 450, 300, 45, 58
    row_count = math.ceil(len(rows) / columns)
    images = []
    for index in range(min(len(video) for _, video in rows)):
        canvas = Image.new('RGB', (columns * width, 50 + row_count * (height + heading) + footer), '#f5f7fa')
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 9), title, font=_font(27, True), fill='#17263a')
        for position, (name, video) in enumerate(rows):
            elapsed, image, phase = video[index]
            x = position % columns * width
            y = 50 + position // columns * (height + heading)
            draw.text((x + 12, y + 6), '%s | %.2f s' % (LABELS[name], elapsed), font=_font(20, True), fill='#334155')
            canvas.paste(image.resize((width, height)), (x, y + heading))
        draw.text((18, canvas.height - 44), 'Movimientos fijos. Sin tiro de balón; no es una prueba de atajadas.', font=_font(22), fill='#334155')
        images.append(canvas)
    # GIF quantizes durations to centiseconds. Preserve 48 ms on average.
    durations = [round((index + 1) * 4.8) * 10 - round(index * 4.8) * 10 for index in range(len(images))]
    images[0].save(output, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False)


def validate(output, seed=71001, hold_s=3., render=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases, pictures, videos = [], {}, {}
    for name in DEFENSE_NAMES + (PREPARATION_NAME,):
        result, frames, animation = validate_case(name, output, seed, hold_s, render)
        cases.append(result)
        pictures[name], videos[name] = frames, animation
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    if render:
        guards = DEFENSE_NAMES[:4]
        jumps = DEFENSE_NAMES[4:]
        _gallery([(name, pictures[name]) for name in guards], output / 'guardias.png', 'Las 4 guardias habilitadas')
        _gallery([(name, pictures[name]) for name in jumps], output / 'saltos.png', 'Los 2 saltos habilitados')
        _gallery([(PREPARATION_NAME, pictures[PREPARATION_NAME])], output / 'preparacion.png', 'Preparación separada de las defensas')
        _animation([(name, videos[name]) for name in DEFENSE_NAMES], output / 'defensas.gif', 'Las 6 defensas originales habilitadas')
        _animation([(PREPARATION_NAME, videos[PREPARATION_NAME])], output / 'preparacion.gif', 'Preparación original: sitDownKeeper', columns=2)
    report = {'validation': VERSION, 'catalog': catalog(), 'seed': seed, 'hold_s': hold_s,
              'executed_on_hardware': False,
              'physics': 'MuJoCo free robot roots, original B-Human targets, servos and contacts',
              'sensor_limitations': 'ideal simulated encoders/IMU and contact FSR; own torso quaternion ideal for original low-level engine',
              'render_only_changes': 'hide striker visual group for the frontal camera; contacts unchanged',
              'ball_velocity_written': False, 'active_root_trajectories_written': False,
              'learned_joint_trajectories': False,
              'limitations': 'No shot, save-rate estimate, hardware equivalence or maximum coverage claim; one initial stance and seed.',
              'recovery_gate': 'stand + upright + torso above 0.28 m + both robots stable for 0.5 s, then keeper upright with both feet loaded for at least 1 s; deadline 28 s',
              'passed': all(case['passed'] for case in cases), 'cases': cases}
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.suffix in ('.png', '.gif', '.jsonl'):
            artifacts.append({'file': path.name, 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    report['artifacts'] = artifacts
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/keeper_defenses/motions')
    parser.add_argument('--seed', type=int, default=71001)
    parser.add_argument('--hold-s', type=float, default=3.)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 1 or not math.isfinite(args.hold_s) or not 2.4 <= args.hold_s <= 3.6:
        parser.error('seed must be 0..2**32-1 and hold-s must be 2.4..3.6')
    report = validate(args.output, args.seed, args.hold_s, not args.no_render)
    print('Passed: %s; summary: %s' % (report['passed'], args.output / 'summary.json'), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
