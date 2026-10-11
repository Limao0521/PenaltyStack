"""Probe six bounded keeper motions and original recovery in free MuJoCo.

This is a simulation execution gate, not a save-rate benchmark or a hardware
qualification. Actual B-Human outputs and measured simulated joint motion are
retained. No active robot-root trajectory or ball velocity is written.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from penalty.keeper_defenses import DEFENSE_MOTIONS, DEFENSE_NAMES
from penalty.rl.motion_world import MotionPhysicalSimulator
from penalty.sim_sensors import load_sensor_model
from penalty.sim_session import SimulationSession
from penalty.skills.keeper_motion import (MOTION_SKILL_VERSION, MotionParameters,
                                         ParameterizedKeeperMotion, motion_parameters)

VALIDATION_VERSION = 'keeper_motion_physical_validation_v1'


def _frame(sim, began):
    output = sim.outputs.get('keeper', {})
    info = output.get('motion_info', {})
    sensors = sim.adapter.frame('keeper')
    request = sim.last_inputs.get('keeper', {}).get('request', {})
    return {'time_s': round(float(sim.data.time)-began, 6),
            'sequence': sim.sequence, 'request_motion': request.get('motion'),
            'request_defense': request.get('block_side'),
            'profile_requested': request.get('keeper_motion_skill'),
            'phase': output.get('phase'), 'keyframe': info.get('keyframe'),
            'mirror': info.get('mirror'),
            'fall_state': output.get('sensor_info', {}).get('fall_state'),
            'output_angles_rad': list(output.get('angles', [])),
            'output_stiffness': list(output.get('stiffness', [])),
            'measured_angles_rad': sensors['angles'],
            'measured_velocity_rad_s': sensors['velocity'],
            'gyro_rad_s': sensors['gyro'], 'fsr_kg': sensors['fsr'],
            'torso_position_m': sim.pose('keeper')[0].tolist(),
            'stable': bool(sim.stable())}


def validate_motion_case(parameters, defense, seed, hold_s=3., render_output=None):
    """Return ``(summary, trace_rows)``; always collect both native children.

    ``parameters=None`` tests the unchanged legacy request. Optional images
    only change rendering visibility and do not alter contacts or state.
    """
    parameters = motion_parameters(parameters)
    if defense not in DEFENSE_NAMES:
        raise ValueError('se esperaba una de las seis defensas')
    if type(seed) is not int or not 0 <= seed <= 2**32-1:
        raise ValueError('seed debe ser entero dentro de 0..2**32-1')
    if not math.isfinite(hold_s) or not 2.4 <= hold_s <= 3.6:
        raise ValueError('hold_s debe estar dentro de 2.4..3.6')
    summary = {'validation': VALIDATION_VERSION, 'defense': defense, 'seed': seed,
               'parameters': None if parameters is None else asdict(parameters),
               'expected_keyframe': DEFENSE_MOTIONS[defense][1],
               'expected_mirror': DEFENSE_MOTIONS[defense][2], 'hold_s': hold_s,
               'executed_on_hardware': False, 'active_root_trajectories_written': False,
               'ball_velocity_written': False}
    sim = MotionPhysicalSimulator(parameters, seed=seed, ball_xy=(1., .8),
                                  model_factory=load_sensor_model)
    session = SimulationSession(sim, record_dir=None, mode='manual', publish_ticks=False)
    processes, records, renderer = [], [], None
    native_matches = 0
    try:
        session.command('role', 'keeper')
        session.command('start')
        processes = [bridge.process for bridge in sim.bridges.values()]
        hello = sim.bridges['keeper'].hello
        summary['runtime_hello'] = hello
        if parameters is not None and MOTION_SKILL_VERSION not in hello.get('capabilities', ()):
            raise RuntimeError('recompila B-Human: falta keeper_motion_skill_v1')
        for role in ('keeper', 'striker'):
            if sim.model.joint(role+'_root_free').type[0] != mujoco.mjtJoint.mjJNT_FREE:
                raise RuntimeError('la prueba requiere raíces libres')
        for _ in range(418):
            session.tick()
            if session.state == 'RUNNING':
                break
        else:
            raise RuntimeError('los robots no se estabilizaron en cinco segundos')
        summary['settle_time_s'] = float(sim.data.time)
        began = float(sim.data.time)
        initial = np.array(sim.adapter.frame('keeper')['angles'])
        max_change, last_active_mirror, active_frames = 0., None, 0
        seen_profile = False
        if render_output is not None:
            from PIL import Image
            render_output = Path(render_output)
            render_output.mkdir(parents=True, exist_ok=True)
            for geom in range(sim.model.ngeom):
                name = sim.model.body(sim.model.geom_bodyid[geom]).name
                if name.startswith('striker_'):
                    sim.model.geom_group[geom] = 5
            option, view = mujoco.MjvOption(), mujoco.MjvCamera()
            option.geomgroup[5] = 0
            view.type = mujoco.mjtCamera.mjCAMERA_FREE
            view.lookat[:] = [-1.10, 0., .30]
            view.distance, view.azimuth, view.elevation = 1.55, 0., -8.
            renderer = mujoco.Renderer(sim.model, height=400, width=600)
            capture_times, next_capture = (0., .36, .72, 1.2, 2.4), 0
        session.command('manual', {'motion': 'block', 'block_side': defense})
        for tick in range(math.ceil(hold_s/sim.dt)+1):
            row = _frame(sim, began)
            profile = sim.outputs['keeper'].get('keeper_motion_skill')
            if profile is not None and not seen_profile:
                row['native_motion_profile'] = profile
                seen_profile = True
            records.append(row)
            max_change = max(max_change, float(np.max(np.abs(
                np.array(row['measured_angles_rad'])-initial))))
            if row['phase'] == 'keyframeMotion' and row['keyframe'] == summary['expected_keyframe']:
                active_frames += 1
                last_active_mirror = row['mirror']
                native_matches += row['mirror'] == summary['expected_mirror']
            if renderer is not None and next_capture < len(capture_times):
                if row['time_s'] + 1e-9 >= capture_times[next_capture]:
                    renderer.update_scene(sim.data, camera=view, scene_option=option)
                    name = '%s_%.2fs.png' % (defense, capture_times[next_capture])
                    Image.fromarray(renderer.render().copy()).save(render_output/name)
                    next_capture += 1
            if tick < math.ceil(hold_s/sim.dt):
                session.tick()
        summary.update(matching_keyframe_frames=native_matches, keyframe_frames=active_frames,
                       max_joint_change_rad=max_change, last_active_mirror=last_active_mirror,
                       motion_executed=(native_matches > 0 and max_change > .1
                                        and last_active_mirror == summary['expected_mirror']),
                       motion_evidence=sim.motion_evidence)
        summary['profile_verified'] = ((parameters is None and sim.motion_evidence is None)
                                       or (parameters is not None and sim.motion_evidence is not None))
        summary['manual_recovery_request_time_s'] = round(float(sim.data.time)-began, 6)
        session.command('manual', {'motion': 'getup'})
        stable_for, recovered_at, stable_hold = 0., None, 0.
        interrupted = []
        while session.penalty_time < 28.-1e-9:
            session.tick()
            row = _frame(sim, began)
            records.append(row)
            upright = (row['fall_state'] == 'upright' and row['phase'] == 'stand'
                       and row['stable'] and row['torso_position_m'][2] > .28)
            if recovered_at is None:
                stable_for = stable_for+sim.dt if upright else 0.
                if stable_for >= .5:
                    recovered_at = row['time_s']
                    session.command('manual', {'motion': 'stand'})
            else:
                still_upright = (row['fall_state'] == 'upright' and row['phase'] == 'stand'
                                 and row['torso_position_m'][2] > .28
                                 and sum(row['fsr_kg'][:4]) > .05
                                 and sum(row['fsr_kg'][4:]) > .05)
                if not still_upright:
                    interrupted.append({'gate_time_s': recovered_at, 'lost_time_s': row['time_s']})
                    stable_for, recovered_at, stable_hold = 0., None, 0.
                else:
                    stable_hold += sim.dt
                    if stable_hold >= 1.:
                        break
        summary.update(recovered_to_stable_stand=recovered_at is not None and stable_hold >= 1.,
                       stable_stand_time_s=recovered_at, upright_confirmation_s=round(stable_hold, 6),
                       interrupted_recovery_attempts=interrupted,
                       end_time_s=round(float(sim.data.time)-began, 6),
                       max_keeper_joint_velocity_rad_s=sim.max_keeper_joint_velocity,
                       profile_request_count=sim.profile_request_count,
                       final_phase=sim.outputs['keeper'].get('phase'),
                       final_fall_state=sim.outputs['keeper'].get('sensor_info', {}).get('fall_state'))
        summary['passed'] = (summary['motion_executed'] and summary['profile_verified']
                             and summary['recovered_to_stable_stand'])
    except Exception as error:
        summary.update(passed=False, error=str(error))
    finally:
        try:
            if renderer is not None:
                renderer.close()
        finally:
            try:
                sim.close()
            finally:
                summary['children_collected'] = all(process.poll() is not None for process in processes)
                summary['passed'] = summary.get('passed', False) and summary['children_collected']
    return summary, records


def validate(parameters, output, seed=78001, render=False, defenses=DEFENSE_NAMES):
    parameters = motion_parameters(parameters)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for defense in defenses:
        summary, rows = validate_motion_case(parameters, defense, seed,
                                            render_output=output if render else None)
        trace = output/(defense+'.frames.jsonl')
        trace.write_bytes(('\n'.join(json.dumps(row, allow_nan=False) for row in rows)+'\n').encode())
        summary.update(trace=trace.name, trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        cases.append(summary)
        print('%s: %s' % (defense, summary['passed']), flush=True)
    report = {'validation': VALIDATION_VERSION, 'motion_skill': MOTION_SKILL_VERSION,
              'parameters': None if parameters is None else asdict(parameters),
              'bounds': ParameterizedKeeperMotion().bounds(), 'seed': seed,
              'physics': 'free roots, original B-Human keyframe engine, servos and physical contacts',
              'sensor_limitations': 'ideal simulated encoders/IMU and contact FSR; no hardware equivalence',
              'recovery_gate': 'stand/upright/stable/torso above 0.28 m for 0.5 s, then both feet loaded for 1 s; 28 s deadline',
              'limitations': 'execution and recovery only; no shot, maximum coverage or save-rate evidence',
              'passed': all(case['passed'] for case in cases), 'cases': cases}
    (output/'summary.json').write_bytes((json.dumps(report, indent=2, ensure_ascii=False,
                                                   allow_nan=False)+'\n').encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, help='JSON with exactly the four parameter fields')
    parser.add_argument('--legacy', action='store_true', help='omit the new motion contract')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/keeper_motion/validation')
    parser.add_argument('--seed', type=int, default=78001)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--defenses', nargs='+', choices=DEFENSE_NAMES, default=list(DEFENSE_NAMES))
    args = parser.parse_args()
    if args.legacy and args.parameters:
        parser.error('--legacy y --parameters son excluyentes')
    params = None if args.legacy else (MotionParameters() if args.parameters is None else
                                      motion_parameters(json.loads(args.parameters.read_text(encoding='utf-8'))))
    report = validate(params, args.output, args.seed, args.render, args.defenses)
    print('Passed: %s; summary: %s' % (report['passed'], args.output/'summary.json'), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
