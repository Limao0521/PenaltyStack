"""Paired duels for keepers v4/v5 on the sensor-camera pipeline (v1).

Same world, striker styles, noise, seeds and camera/sensor pipeline as
orientation_benchmark (keeper_sensor_camera_v2), so scripted/v2/v3 results
are reproduced and new keepers are compared on identical kicks. `observer`
never commits: its traces give each seed's uncontaminated perception stream.
Truth is read only for evaluation fields, never by perception or policy.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import json
import math
import time
import traceback

import numpy as np

from penalty.keeper_pose_vision import VisionPoseKeeper
from penalty.perception import local
from penalty.rl.contracts import KeeperAction, StrikerAction, clip_observation
from penalty.rl.duel import make_player, player_spec
from penalty.rl.keeper_benchmark import STYLES, VariedStrikerPolicy, aggregate, compare_styles
from penalty.rl.orientation_benchmark import measurements, perception_summary
from penalty.rl.orientation_contracts import from_vision_state
from penalty.rl.policies.baselines import ScriptedKeeperPolicy
from penalty.rl.vision_benchmark import base_action
from penalty.rl.vision_keeper import VisionKeeperPolicy
from penalty.rl.wrappers import PerceptionNoise
from penalty.sim_orientation import PIPELINE_VERSION, SensorBHumanKeeperCamera, SensorCameraDuelWorld

BENCHMARK_VERSION = 'keeper_fusion_benchmark_v1'
# Name -> (policy version, uses body yaw, ball logic).
KEEPERS = {
    'scripted': ('scripted_keeper (project FSM)', False, 'bearing gate'),
    'vision': ('vision_keeper_v2', False, 'v2'),
    'pose': ('vision_keeper_v3', True, 'v2'),
    'pose_v4': ('vision_keeper_v4', True, 'v2'),
    'ball_v5': ('vision_keeper_v5_ball', False, 'v5'),
    'full_v5': ('vision_keeper_v5', True, 'v5'),
    'observer': ('observer (never commits)', False, 'none'),
}


class ObserverKeeper:
    """Tracks with the same head controller and never requests a guard."""
    decision = {'reason': 'observer'}

    def __call__(self, state):
        return {'block_side': 0, 'trigger': 0., 'amplitude': .5}


def make_keeper(name):
    if name == 'scripted':
        return ScriptedKeeperPolicy()
    if name == 'vision':
        return VisionKeeperPolicy()
    if name == 'pose':
        return VisionPoseKeeper()
    if name in ('pose_v4', 'ball_v5', 'full_v5'):
        from penalty.keeper_fusion import FusionKeeper
    if name == 'pose_v4':
        return FusionKeeper.v4()
    if name == 'ball_v5':
        return FusionKeeper.v5(use_pose=False)
    if name == 'full_v5':
        return FusionKeeper.v5(use_pose=True)
    if name == 'observer':
        return ObserverKeeper()
    raise ValueError('portero desconocido: ' + str(name))


def _round(values, digits=4):
    return [None if v is None else round(float(v), digits) for v in values]


def compact_frame(world, state, action, decision):
    """Small per-frame trace; `truth` is evaluation-only and never fed back."""
    b, o, h = state['ball'], state['opponent'], state['opponent_orientation']
    vc, pc = b['velocity_covariance'], b.get('position_covariance', ((0., 0.), (0., 0.)))
    odometry = state.get('sensors', {}).get('odometry', (0., 0., 0.))
    xy, yaw = world._ground_pose('keeper')
    rival, rival_yaw = world._ground_pose('striker')
    ball_world = world._ball_xy()
    return {'t': round(state['time_s'], 4), 'cap': round(state['capture_s'], 4),
            'b': _round(b['position'], 5)+_round(b['velocity'], 5)+_round((vc[0][0], vc[1][1], vc[0][1],
                         pc[0][0], pc[1][1]), 8)+_round((b['age_s'],))+[int(b['visible']), int(b['position_valid']),
                                                                     int(b['velocity_valid'])],
            'o': _round((*o['position'], o['age_s'], o['yaw'], o['yaw_variance'], o['yaw_age_s'],
                         o['yaw_capture_s'], h.get('candidate_yaw'), h.get('match_distance'),
                         h.get('angular_sigma_rad'))) + [int(o['position_valid']), int(o['yaw_valid']),
                                                         o.get('track_id'), h.get('reason')],
            'od': _round(odometry),
            'a': [int(action.block_side), round(float(action.trigger), 3)],
            'd': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in decision.items()
                  if k in ('reason', 'side', 'crossing_y', 'time_to_line', 'pose_y', 'pose_sigma',
                           'fused_y', 'fused_sigma', 'ball_sigma', 'hold_reason', 'pose_used', 'ball_side',
                           'holds', 'track_points', 'pose_prior_used', 'pose_prior_frames')},
            'truth': _round((*local(ball_world, (*xy, yaw)), *local(rival, (*xy, yaw)),
                             math.remainder(rival_yaw-yaw, 2*math.pi), *ball_world, *xy, yaw))}


def camera_episode(world, camera, striker, keeper_name, seed, frames=None):
    rng = np.random.default_rng(seed)
    striker_obs, _, _ = world.reset(rng, {})
    camera.reset()
    keeper = make_keeper(keeper_name)
    if hasattr(striker, 'reset'):
        striker.reset()
    noise, noise_rng = PerceptionNoise(), np.random.default_rng([seed, 0])
    result, steps, pose_uses, samples = None, 0, 0, []
    while result is None:
        state = camera.observe()
        observation = from_vision_state(state)
        if keeper_name == 'scripted':
            action = base_action(keeper, observation)
        elif keeper_name == 'vision':
            action = KeeperAction.from_normalized(keeper(observation.to_array()[:26]))
        else:
            action = KeeperAction(**keeper(state))
        samples.append(measurements(world, state))
        decision = dict(getattr(keeper, 'decision', {}))
        pose_uses += bool(decision.get('pose_prior_used') or decision.get('pose_used'))
        if frames is not None:
            frames.append(compact_frame(world, state, action, decision))
        striker_action = StrikerAction.from_normalized(striker(noise(clip_observation(striker_obs.to_array()), noise_rng)))
        striker_obs, _, result, _ = world.duel_step(striker_action, action)
        steps += 1
    return dict(world.record(), seed=seed, steps=steps, perception=PIPELINE_VERSION,
                pose_prior_used=bool(pose_uses), visible_frames=sum(f['ball_visible'] for f in samples),
                velocity_frames=sum(f['ball_velocity_valid'] for f in samples),
                position_error_m=[f['evaluation_only_ball_error_m'] for f in samples
                                  if 'evaluation_only_ball_error_m' in f],
                velocity_error_m_s=[], opponent_measurements=samples,
                keeper_decision=dict(getattr(keeper, 'decision', {})))


def _job(job):
    world, camera = SensorCameraDuelWorld(ball_jitter=.05, start_jitter=(.05, .1)), None
    records = []
    try:
        for seed in job['seeds']:
            world._ensure_session()
            if camera is not None and camera.model is not world.sim.model:
                camera.close()
                camera = None
            if camera is None:
                camera = SensorBHumanKeeperCamera(world)
            striker = (make_player(job['trained'][job['style']], 'striker') if job['style'] in ('v1', 'v2')
                       else VariedStrikerPolicy(job['style'], seed))
            frames = [] if job['frames'] else None
            try:
                record = camera_episode(world, camera, striker, job['keeper'], seed, frames)
            except (RuntimeError, ValueError, OSError) as error:
                record = {'result': 'sim_error', 'seed': seed, 'steps': 0, 'sim_error': str(error),
                          'visible_frames': 0, 'velocity_frames': 0, 'position_error_m': [],
                          'velocity_error_m_s': [], 'opponent_measurements': []}
                world.close()
            if frames is not None:
                record['frames'] = frames
            records.append(record)
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            world.close()
    return records


def _failed(job, error):
    """Job-level failure: every seed counts as sim_error, with the traceback."""
    return [{'result': 'sim_error', 'seed': seed, 'steps': 0, 'sim_error': error, 'visible_frames': 0,
             'velocity_frames': 0, 'position_error_m': [], 'velocity_error_m_s': [], 'opponent_measurements': []}
            for seed in job['seeds']]


def _run_jobs(jobs, workers, finish, progress=None, max_restarts=6):
    """Run jobs; a crashed worker (broken pool) restarts the pool for unfinished jobs.

    An exception raised by a job is retried once and then recorded as
    sim_error for its seeds, so one failure never discards the experiment.
    """
    pending, attempts, restarts = dict(enumerate(jobs)), {}, 0
    while pending:
        if workers <= 1:
            for index, job in list(pending.items()):
                try:
                    records = _job(job)
                except Exception:
                    records = _failed(job, traceback.format_exc())
                finish(job, records)
                del pending[index]
            return
        broken = False
        with ProcessPoolExecutor(min(workers, len(pending))) as pool:
            futures = {pool.submit(_job, job): index for index, job in pending.items()}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    records = future.result()
                except BrokenProcessPool:
                    broken = True
                    continue
                except Exception:
                    attempts[index] = attempts.get(index, 0)+1
                    if attempts[index] < 2:
                        continue
                    records = _failed(pending[index], traceback.format_exc())
                finish(pending[index], records)
                del pending[index]
        if pending and broken:
            restarts += 1
            if progress:
                progress('Un worker terminó inesperadamente; reinicio %d con %d trabajos pendientes' % (
                    restarts, len(pending)), flush=True)
            if restarts > max_restarts:
                for index, job in list(pending.items()):
                    finish(job, _failed(job, 'worker pool broke %d times' % restarts))
                    del pending[index]


def benchmark(keepers, episodes=50, seed=70001, workers=4, styles=STYLES, frames=False, frames_path=None,
              progress=print):
    if (episodes < 1 or workers < 1 or not keepers or len(set(keepers)) != len(keepers) or
            any(k not in KEEPERS for k in keepers) or not styles or len(set(styles)) != len(styles) or
            any(s not in STYLES for s in styles)):
        raise ValueError('episodios/workers positivos y porteros/estilos conocidos y distintos requeridos')
    trained = {name: player_spec('models/rl/striker_'+name)[1] for name in ('v1', 'v2') if name in styles}
    seeds = [seed*1_000_000+i for i in range(episodes)]
    jobs = [{'style': style, 'keeper': keeper, 'trained': trained, 'seeds': seeds[i:i+5], 'frames': frames}
            for keeper in keepers for style in styles for i in range(0, episodes, 5)]
    groups = {keeper: {style: [] for style in styles} for keeper in keepers}
    start = time.perf_counter()
    stream = open(frames_path, 'w', encoding='utf-8') if frames_path else None
    done = 0

    def finish(job, records):
        nonlocal done
        done += 1
        for record in records:
            trace = record.pop('frames', None)
            if stream is not None and trace is not None:
                stream.write(json.dumps({'keeper': job['keeper'], 'style': job['style'], 'seed': record['seed'],
                                         'record': {k: v for k, v in record.items() if k != 'opponent_measurements'},
                                         'frames': trace}, allow_nan=False)+'\n')
        groups[job['keeper']][job['style']].extend(records)
        if progress:
            progress('[%d/%d] %s / %s: %d/%d atajadas (%.0f s)' % (
                done, len(jobs), job['keeper'], job['style'], sum(r['result'] == 'save' for r in records),
                len(records), time.perf_counter()-start), flush=True)
        if stream is not None:
            stream.flush()
    try:
        _run_jobs(jobs, workers, finish, progress)
    finally:
        if stream is not None:
            stream.close()
    for gs in groups.values():
        for rs in gs.values():
            rs.sort(key=lambda r: r['seed'])
    frames_all = [f for gs in groups.values() for rs in gs.values() for r in rs for f in r['opponent_measurements']]
    for gs in groups.values():
        for rs in gs.values():
            for r in rs:
                r.pop('opponent_measurements', None)
    reference = 'full_v5' if 'full_v5' in groups else keepers[0]
    return {'benchmark': BENCHMARK_VERSION, 'camera_pipeline': PIPELINE_VERSION, 'seed': seed,
            'episode_seeds': [seeds[0], seeds[-1]], 'episodes_per_style': episodes, 'styles': list(styles),
            'keepers': {k: KEEPERS[k] for k in keepers}, 'wall_s': time.perf_counter()-start,
            'perception': 'RGB cameras, original B-Human robot boxes, visual reference body yaw; own encoder/IMU/FSR sensors',
            'sensor_limitations': 'ideal simulated sensor values; estimated gyro/gravity attitude and contact odometry; '
                                  'nominal camera/sole geometry; wall inference does not advance physics',
            'striker_perception': 'scene privileged plus PerceptionNoise v1 (keeper sensors only)',
            'summary': {name: aggregate(gs) for name, gs in groups.items() if name != 'observer'},
            'perception_summary': perception_summary(frames_all),
            'comparisons': {'%s_vs_%s' % (a, b): compare_styles(groups[a], groups[b])
                            for a, b in (('vision', 'scripted'), ('pose', 'vision'), ('pose_v4', 'vision'),
                                         ('ball_v5', 'vision'), ('full_v5', 'ball_v5'), ('full_v5', 'vision'),
                                         ('full_v5', 'scripted'), ('pose_v4', 'pose'))
                            if a in groups and b in groups},
            'records': groups}
