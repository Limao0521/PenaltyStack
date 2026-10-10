"""Paired B-Human-motion duels with real rendered keeper pixels (v1).

Striker retains the earlier privileged/noisy tactical observations. Both
keepers see the same new pixel detector, calibration, odometry and tracker.
No target, contact event, opponent orientation or scene ball pose is passed
to the keeper. Truth is read separately only for error reporting.
"""
from concurrent.futures import ProcessPoolExecutor
import math
import time

import numpy as np

from penalty.rl.contracts import KeeperAction, StrikerAction, clip_observation
from penalty.rl.duel import make_player, player_spec
from penalty.rl.keeper_benchmark import STYLES, VariedStrikerPolicy, aggregate, compare_styles
from penalty.rl.policies.baselines import ScriptedKeeperPolicy
from penalty.rl.vision_contracts import from_vision_state
from penalty.rl.vision_keeper import VisionKeeperPolicy
from penalty.rl.wrappers import PerceptionNoise
from penalty.sim_vision import CameraDuelWorld, SimulatedKeeperCamera
from penalty.perception import local

BENCHMARK_VERSION = 'keeper_camera_benchmark_v1'


def base_action(policy, observation):
    # Base consumes v1 ball fields; apply the same freshness/missing-velocity
    # eligibility as v2, because v1 cannot express an unmeasured velocity.
    if not (observation.ball_visible and observation.ball_position_valid and
            observation.ball_velocity_valid and observation.ball_age <= .25):
        return KeeperAction(0, 0., .5)
    return KeeperAction.from_normalized(policy(observation.to_array()[:19]))


def camera_episode(world, camera, striker, keeper_name, seed, trace=None, snapshot=None):
    rng = np.random.default_rng(seed)
    striker_obs, _, _ = world.reset(rng, {})
    camera.reset()
    keeper = VisionKeeperPolicy() if keeper_name == 'vision' else ScriptedKeeperPolicy()
    if hasattr(striker, 'reset'):
        striker.reset()
    noise, noise_rng = PerceptionNoise(), np.random.default_rng([seed, 0])
    result, steps = None, 0
    errors, velocity_errors, visible, valid_velocity = [], [], 0, 0
    while result is None:
        state = camera.observe()
        observation = from_vision_state(state)
        if keeper_name == 'vision':
            action = KeeperAction.from_normalized(keeper(observation.to_array()))
        else:
            action = base_action(keeper, observation)
        # Evaluation truth never enters the observation or the policy.
        xy, yaw = world._ground_pose('keeper')
        truth = local(world._ball_xy(), (*xy, yaw))
        if observation.ball_position_valid and observation.ball_visible:
            visible += 1
            errors.append(math.hypot(observation.ball_x-truth[0], observation.ball_y-truth[1]))
        if observation.ball_velocity_valid:
            valid_velocity += 1
            ball_velocity = world.sim.data.qvel[world.sim.ball_qvel:world.sim.ball_qvel+2]
            truth_velocity = local(ball_velocity, (0, 0, yaw))
            velocity_errors.append(math.hypot(observation.ball_vx-truth_velocity[0],
                                             observation.ball_vy-truth_velocity[1]))
        if trace is not None:
            trace.append({'seed': seed, 'vision': state, 'action': vars(action),
                          'decision': dict(getattr(keeper, 'decision', {})),
                          'evaluation_only_ball_position': truth})
        if steps == 0 and snapshot is not None:
            from PIL import Image
            Image.fromarray(camera.rgb).save(snapshot)
        striker_action = StrikerAction.from_normalized(striker(noise(
            clip_observation(striker_obs.to_array()), noise_rng)))
        striker_obs, _, result, _ = world.duel_step(striker_action, action)
        steps += 1
    return dict(world.record(), seed=seed, steps=steps, perception='simulated_camera',
                pose_prior_used=False, visible_frames=visible, velocity_frames=valid_velocity,
                position_error_m=errors, velocity_error_m_s=velocity_errors,
                keeper_decision=dict(getattr(keeper, 'decision', {})))


def _job(job):
    world = CameraDuelWorld(ball_jitter=.05, start_jitter=(.05, .1))
    camera = None
    records = []
    try:
        for seed in job['seeds']:
            # Build renderer after the first model exists; reset per episode.
            world._ensure_session()
            if camera is not None and camera.model is not world.sim.model:
                camera.close()
                camera = None
            if camera is None:
                camera = SimulatedKeeperCamera(world)
            striker = (make_player(job['trained'][job['style']], 'striker') if job['style'] in ('v1', 'v2')
                       else VariedStrikerPolicy(job['style'], seed))
            try:
                records.append(camera_episode(world, camera, striker, job['keeper'], seed))
            except (RuntimeError, ValueError, OSError) as error:
                records.append({'result': 'sim_error', 'seed': seed, 'steps': 0,
                                'sim_error': str(error), 'perception': 'simulated_camera',
                                'visible_frames': 0, 'velocity_frames': 0,
                                'position_error_m': [], 'velocity_error_m_s': []})
                world.close()
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            world.close()  # Renderer failure must still collect both native children.
    return records


def benchmark(episodes=5, seed=65001, workers=2, styles=STYLES):
    if episodes < 1 or workers < 1 or not styles or any(s not in STYLES for s in styles):
        raise ValueError('positive episodes/workers and known styles required')
    trained = {name: player_spec('models/rl/striker_'+name)[1] for name in ('v1', 'v2') if name in styles}
    seeds = [seed*1_000_000+i for i in range(episodes)]
    jobs = [{'style': style, 'keeper': keeper, 'trained': trained, 'seeds': seeds[i:i+5]}
            for keeper in ('vision', 'scripted') for style in styles for i in range(0, episodes, 5)]
    groups = {name: {style: [] for style in styles} for name in ('vision', 'scripted')}
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
    measurements = {}
    for name, styles_records in groups.items():
        records = [r for rs in styles_records.values() for r in rs]
        positions = [v for r in records for v in r['position_error_m']]
        velocities = [v for r in records for v in r['velocity_error_m_s']]
        measurements[name] = {'visible_fraction': sum(r['visible_frames'] for r in records)/max(1, sum(r['steps'] for r in records)),
                              'position_error_median_m': float(np.median(positions)) if positions else None,
                              'position_error_p95_m': float(np.quantile(positions, .95)) if positions else None,
                              'velocity_error_median_m_s': float(np.median(velocities)) if velocities else None,
                              'velocity_error_p95_m_s': float(np.quantile(velocities, .95)) if velocities else None}
    return {'benchmark': BENCHMARK_VERSION, 'seed': seed, 'episodes_per_style': episodes,
            'styles': list(styles), 'wall_s': time.perf_counter()-start,
            'perception': 'keeper: rendered RGB lower head camera; orange detector; calibrated ground projection; odometry tracker',
            'sensor_limitations': 'ideal simulated extrinsics/odometry/IMU; nominal intrinsics; no physical calibration',
            'opponent_position_valid': False, 'opponent_yaw_valid': False,
            'striker_perception': 'scene privileged plus PerceptionNoise v1',
            'summary': {name: aggregate(gs) for name, gs in groups.items()},
            'measurement_errors': measurements, 'comparison': compare_styles(groups['vision'], groups['scripted']),
            'records': groups}
