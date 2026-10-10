"""Paired v3/v2/base duels using RGB and own simulated sensors (v1)."""
from concurrent.futures import ProcessPoolExecutor
import math
import time

import numpy as np

from penalty.keeper_pose_vision import VisionPoseKeeper
from penalty.opponent_orientation import wrap
from penalty.perception import local
from penalty.rl.contracts import KeeperAction, StrikerAction, clip_observation
from penalty.rl.duel import make_player, player_spec
from penalty.rl.keeper_benchmark import STYLES, VariedStrikerPolicy, aggregate, compare_styles
from penalty.rl.orientation_contracts import from_vision_state
from penalty.rl.policies.baselines import ScriptedKeeperPolicy
from penalty.rl.vision_benchmark import base_action
from penalty.rl.vision_keeper import VisionKeeperPolicy
from penalty.rl.wrappers import PerceptionNoise
from penalty.sim_orientation import SensorCameraDuelWorld, SensorBHumanKeeperCamera, PIPELINE_VERSION

BENCHMARK_VERSION = 'keeper_sensor_orientation_benchmark_v1'


def measurements(world, state):
    """Truth is read here, outside observation, orientation and policy."""
    xy, yaw = world._ground_pose('keeper')
    rival, rival_yaw = world._ground_pose('striker')
    truth = local(world._ball_xy(), (*xy, yaw))
    o, b, heading = state['opponent'], state['ball'], state['opponent_orientation']
    item = {'capture_s': state['capture_s'], 'ball_visible': b['visible'],
            'ball_velocity_valid': b['velocity_valid'], 'opponent_valid': o['position_valid'],
            'yaw_valid': o['yaw_valid'], 'yaw_reason': heading['reason'],
            'yaw_processing_ms': heading['processing_ms'], 'yaw_variance': o['yaw_variance'],
            'yaw_age_s': o['yaw_age_s'], 'match_distance': heading.get('match_distance'),
            'angular_sigma_rad': heading.get('angular_sigma_rad'),
            'evaluation_only_ball_position': truth,
            'evaluation_only_rival_yaw': wrap(rival_yaw-yaw)}
    if b['position_valid'] and b['visible']:
        item['evaluation_only_ball_error_m'] = math.hypot(b['position'][0]-truth[0], b['position'][1]-truth[1])
    if o['position_valid']:
        position = local(rival, (*xy, yaw))
        item['evaluation_only_rival_error_m'] = math.hypot(o['position'][0]-position[0], o['position'][1]-position[1])
    if o['yaw_valid']:
        item['evaluation_only_yaw_error_deg'] = math.degrees(abs(wrap(o['yaw']-item['evaluation_only_rival_yaw'])))
    return item


def camera_episode(world, camera, striker, keeper_name, seed, trace=None, on_frame=None):
    rng = np.random.default_rng(seed)
    striker_obs, _, _ = world.reset(rng, {})
    camera.reset()
    keeper = (VisionPoseKeeper() if keeper_name == 'pose' else VisionKeeperPolicy() if keeper_name == 'vision'
              else ScriptedKeeperPolicy())
    if hasattr(striker, 'reset'):
        striker.reset()
    noise, noise_rng = PerceptionNoise(), np.random.default_rng([seed, 0])
    result, steps, pose_uses = None, 0, 0
    samples = []
    while result is None:
        state = camera.observe()
        observation = from_vision_state(state)
        if keeper_name == 'pose':
            action = KeeperAction(**keeper(state))
        elif keeper_name == 'vision':
            action = KeeperAction.from_normalized(keeper(observation.to_array()[:26]))
        else:
            action = base_action(keeper, observation)
        sample = measurements(world, state)
        samples.append(sample)
        decision = dict(getattr(keeper, 'decision', {}))
        pose_uses += bool(decision.get('pose_prior_used'))
        if trace is not None:
            trace.append({'seed': seed, 'vision': state, 'action': vars(action), 'decision': decision,
                          'evaluation_only': sample, 'executed_on_hardware': False})
        if on_frame:
            on_frame(world, camera, state, action, decision)
        striker_action = StrikerAction.from_normalized(striker(noise(clip_observation(striker_obs.to_array()), noise_rng)))
        striker_obs, _, result, _ = world.duel_step(striker_action, action)
        steps += 1
    return dict(world.record(), seed=seed, steps=steps, perception=PIPELINE_VERSION,
                pose_prior_used=bool(pose_uses), visible_frames=sum(f['ball_visible'] for f in samples),
                velocity_frames=sum(f['ball_velocity_valid'] for f in samples),
                position_error_m=[f['evaluation_only_ball_error_m'] for f in samples if 'evaluation_only_ball_error_m' in f],
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
            try:
                records.append(camera_episode(world, camera, striker, job['keeper'], seed))
            except (RuntimeError, ValueError, OSError) as error:
                records.append({'result': 'sim_error', 'seed': seed, 'steps': 0, 'sim_error': str(error),
                                'visible_frames': 0, 'velocity_frames': 0, 'position_error_m': [],
                                'velocity_error_m_s': [], 'opponent_measurements': []})
                world.close()
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            world.close()
    return records


def perception_summary(frames):
    errors = [f['evaluation_only_yaw_error_deg'] for f in frames if f['yaw_valid']]
    ball_errors = [f['evaluation_only_ball_error_m'] for f in frames if 'evaluation_only_ball_error_m' in f]
    return {'frames': len(frames), 'position_valid_frames': sum(f['opponent_valid'] for f in frames),
            'yaw_valid_frames': len(errors), 'yaw_valid_fraction': len(errors)/max(1, len(frames)),
            'yaw_error_median_deg': float(np.median(errors)) if errors else None,
            'yaw_error_p95_deg': float(np.quantile(errors, .95)) if errors else None,
            'front_back_errors_over_90deg': sum(e > 90 for e in errors),
            'ball_error_median_m': float(np.median(ball_errors)) if ball_errors else None,
            'ball_error_p95_m': float(np.quantile(ball_errors, .95)) if ball_errors else None,
            'yaw_processing_median_ms': float(np.median([f['yaw_processing_ms'] for f in frames])) if frames else None,
            'reasons': {reason: sum(f['yaw_reason'] == reason for f in frames) for reason in sorted({f['yaw_reason'] for f in frames})}}


def benchmark(episodes=2, seed=68001, workers=4, styles=STYLES):
    if episodes < 1 or workers < 1 or not styles or len(set(styles)) != len(styles) or any(s not in STYLES for s in styles):
        raise ValueError('positive episodes/workers and distinct known styles required')
    trained = {name: player_spec('models/rl/striker_'+name)[1] for name in ('v1', 'v2') if name in styles}
    seeds = [seed*1_000_000+i for i in range(episodes)]
    jobs = [{'style': style, 'keeper': keeper, 'trained': trained, 'seeds': seeds[i:i+5]}
            for keeper in ('pose', 'vision', 'scripted') for style in styles for i in range(0, episodes, 5)]
    groups = {keeper: {style: [] for style in styles} for keeper in ('pose', 'vision', 'scripted')}
    start = time.perf_counter()
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        results = pool.map(_job, jobs) if pool else map(_job, jobs)
        for job, records in zip(jobs, results):
            groups[job['keeper']][job['style']].extend(records)
            print('%s / %s: %d/%d atajadas' % (job['keeper'], job['style'],
                   sum(r['result'] == 'save' for r in records), len(records)), flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    all_frames = [f for gs in groups.values() for records in gs.values() for r in records for f in r['opponent_measurements']]
    return {'benchmark': BENCHMARK_VERSION, 'camera_pipeline': PIPELINE_VERSION, 'seed': seed,
            'episodes_per_style': episodes, 'styles': list(styles), 'wall_s': time.perf_counter()-start,
            'perception': 'RGB cameras, original B-Human robot boxes, visual reference body yaw; own encoder/IMU/FSR sensors',
            'sensor_limitations': 'ideal simulated sensor values; estimated gyro/gravity attitude and contact odometry; nominal camera/sole geometry; wall inference does not advance physics',
            'striker_perception': 'scene privileged plus PerceptionNoise v1 (keeper sensors only)',
            'orientation_model': 'nao_visual_yaw_v1, simulated appearance only, no physical NAO validation',
            'summary': {name: aggregate(gs) for name, gs in groups.items()},
            'perception_summary': perception_summary(all_frames),
            'comparison': {name: compare_styles(groups['pose'], groups[name]) for name in ('vision', 'scripted')},
            'records': groups}
