"""Versioned camera benchmark with the actual B-Human robot detector enabled."""
from concurrent.futures import ProcessPoolExecutor
import math
import time

import numpy as np

from penalty.perception import local
from penalty.rl.duel import make_player, player_spec
from penalty.rl.keeper_benchmark import STYLES, VariedStrikerPolicy, aggregate, compare_styles
from penalty.rl.vision_benchmark import camera_episode
from penalty.sim_opponent_vision import BHumanCameraDuelWorld, BHumanKeeperCamera

BENCHMARK_VERSION = 'keeper_bhuman_opponent_benchmark_v1'


class MeasurementCamera:
    """Evaluation truth stays outside the camera observation and policy."""
    def __init__(self, camera, world):
        self.camera, self.world = camera, world
        self.measurements = []

    @property
    def rgb(self):
        return self.camera.rgb

    def reset(self):
        self.camera.reset()
        self.measurements = []

    def observe(self):
        state = self.camera.observe()
        upper, lower = state['opponent_detector_upper'], state['opponent_detector_lower']
        item = {'capture_s': state['capture_s'], 'ball_visible': state['ball']['visible'],
                'opponent_valid': state['opponent']['position_valid'],
                'yaw_valid': state['opponent']['yaw_valid'],
                'boxes': state['opponent_detector']['boxes'], 'upper_boxes': upper['boxes'], 'lower_boxes': lower['boxes'],
                'max_confidence': upper['max_confidence'],
                'inference_ms': upper['inference_ms'], 'lower_processing_ms': lower['inference_ms'],
                'selected_method': state['opponent_detector']['method'],
                'reason': state['opponent_detector']['reason']}
        if item['opponent_valid']:
            own_xy, yaw = self.world._ground_pose('keeper')
            rival_xy, _ = self.world._ground_pose('striker')
            truth = local(rival_xy, (*own_xy, yaw))
            item['evaluation_only_position_error_m'] = math.hypot(
                state['opponent']['position'][0]-truth[0], state['opponent']['position'][1]-truth[1])
        self.measurements.append(item)
        return state


def _job(job):
    world, camera = BHumanCameraDuelWorld(ball_jitter=.05, start_jitter=(.05, .1)), None
    records = []
    try:
        for seed in job['seeds']:
            world._ensure_session()
            if camera is not None and camera.model is not world.sim.model:
                camera.close()
                camera = None
            if camera is None:
                camera = BHumanKeeperCamera(world)
            striker = (make_player(job['trained'][job['style']], 'striker') if job['style'] in ('v1', 'v2')
                       else VariedStrikerPolicy(job['style'], seed))
            measured = MeasurementCamera(camera, world)
            try:
                record = camera_episode(world, measured, striker, job['keeper'], seed)
                record['opponent_measurements'] = measured.measurements
                record['camera_pipeline'] = 'keeper_bhuman_camera_v1'
                records.append(record)
            except (RuntimeError, ValueError, OSError) as error:
                records.append({'result': 'sim_error', 'seed': seed, 'steps': 0, 'sim_error': str(error),
                                'visible_frames': 0, 'velocity_frames': 0, 'position_error_m': [],
                                'velocity_error_m_s': [], 'opponent_measurements': measured.measurements})
                world.close()
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            world.close()
    return records


def benchmark(episodes=5, seed=66001, workers=2, styles=STYLES):
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
    frames = [f for styles_records in groups.values() for records in styles_records.values()
              for record in records for f in record['opponent_measurements']]
    errors = [f['evaluation_only_position_error_m'] for f in frames if 'evaluation_only_position_error_m' in f]
    return {'benchmark': BENCHMARK_VERSION, 'seed': seed, 'episodes_per_style': episodes,
            'styles': list(styles), 'wall_s': time.perf_counter()-start,
            'perception': 'lower pixels/ball tracker + upper pixels/original B-Human robot core/ONNX',
            'opponent_detector_active': True, 'pose_prior_used': False,
            'sensor_limitations': 'ideal simulated own transforms/odometry/IMU; nominal intrinsics; wall inference does not delay simulated time',
            'opponent_measurements': {'frames': len(frames), 'frames_with_boxes': sum(bool(f['boxes']) for f in frames),
                'upper_box_frames': sum(bool(f['upper_boxes']) for f in frames),
                'lower_box_frames': sum(bool(f['lower_boxes']) for f in frames),
                'position_valid_frames': sum(f['opponent_valid'] for f in frames),
                'yaw_valid_frames': sum(f['yaw_valid'] for f in frames),
                'max_confidence': max((f['max_confidence'] for f in frames), default=0.),
                'inference_median_ms': float(np.median([f['inference_ms'] for f in frames])) if frames else None,
                'lower_processing_median_ms': float(np.median([f['lower_processing_ms'] for f in frames])) if frames else None,
                'position_error_median_m': float(np.median(errors)) if errors else None},
            'striker_perception': 'scene privileged plus PerceptionNoise v1',
            'summary': {name: aggregate(gs) for name, gs in groups.items()},
            'comparison': compare_styles(groups['vision'], groups['scripted']), 'records': groups}
