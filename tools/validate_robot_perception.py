"""Reproducible head-camera detection probe, with evaluation truth kept apart."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from penalty.perception import local
from penalty.sim_opponent_vision import BHumanCameraDuelWorld, BHumanKeeperCamera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--seed', type=int, default=66002)
    parser.add_argument('--reference-image', type=Path,
                        help='optional B-Human Resolution.png montage; only network boxes are evaluated')
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError('at least two frames required')
    args.output.mkdir(parents=True, exist_ok=True)
    world, camera = BHumanCameraDuelWorld(), None
    trace, errors, timings, lower_timings = [], [], [], []
    snapshot_saved, reference = False, None
    try:
        world.reset(np.random.default_rng(args.seed), {})
        camera = BHumanKeeperCamera(world)
        for i in range(args.frames):
            state = camera.observe()
            # Evaluation labels are never returned to the observation/policy.
            own_xy, own_yaw = world._ground_pose('keeper')
            rival_xy, _ = world._ground_pose('striker')
            truth = local(rival_xy, (*own_xy, own_yaw))
            error = (math.hypot(state['opponent']['position'][0]-truth[0],
                               state['opponent']['position'][1]-truth[1])
                     if state['opponent']['position_valid'] else None)
            if error is not None:
                errors.append(error)
            timings.append(state['opponent_detector_upper']['inference_ms'])
            lower_timings.append(state['opponent_detector_lower']['inference_ms'])
            trace.append({'vision': state, 'evaluation_only_rival_position_m': truth,
                          'evaluation_only_position_error_m': error, 'executed_on_hardware': False})
            if state['opponent']['position_valid'] and not snapshot_saved:
                Image.fromarray(camera.rgb).save(args.output / 'lower_camera.png')
                Image.fromarray(camera.upper_rgb).save(args.output / 'upper_camera.png')
                snapshot_saved = True
            world._advance(until=lambda: world.session.penalty_time >= (i+1)*.12)
        runtime = camera.detector.hello
        if args.reference_image:
            data = args.reference_image.read_bytes()
            with Image.open(args.reference_image) as original:
                if original.size != (1234, 463):
                    raise ValueError('expected original B-Human Resolution.png montage size')
                rgb = original.convert('RGB').crop((618, 0, 1234, 463)).resize((640, 480))
            reference = {'source_url': 'https://docs.b-human.de/coderelease2023/images/perception/robotdetection/Resolution.png',
                         'input_sha256': hashlib.sha256(data).hexdigest(), 'crop': [618, 0, 1234, 463],
                         'resize': [640, 480], 'calibration': 'none: image-only network check, no range claim',
                         'detections': camera.detector.detect(rgb.tobytes(), 640, 480, 0.)}
    finally:
        try:
            if camera:
                camera.close()
        finally:
            world.close()
    trace_path = args.output / 'trace.jsonl'
    trace_path.write_bytes(('\n'.join(json.dumps(row, allow_nan=False) for row in trace)+'\n').encode('utf-8'))
    source_files = ['native/bhuman_vision/main.cpp', 'native/bhuman_vision/CMakeLists.txt',
                    'tools/build_robot_perception.py', 'penalty/robot_detector.py',
                    'penalty/opponent_perception.py', 'penalty/sim_opponent_vision.py']
    report = {'validation': 'bhuman_opponent_pixel_probe_v1', 'seed': args.seed,
              'frames': len(trace), 'capture_interval_s': .12, 'scene': 'existing sim/scene.xml',
              'opponent_valid_frames': len(errors),
              'upper_box_frames': sum(bool(row['vision']['opponent_detector_upper']['boxes']) for row in trace),
              'lower_box_frames': sum(bool(row['vision']['opponent_detector_lower']['boxes']) for row in trace),
              'yaw_valid_frames': sum(row['vision']['opponent']['yaw_valid'] for row in trace),
              'position_error_median_m': float(np.median(errors)) if errors else None,
              'position_error_p95_m': float(np.quantile(errors, .95)) if errors else None,
              'upper_inference_median_ms': float(np.median(timings)),
              'lower_processing_median_ms': float(np.median(lower_timings)),
              'runtime': runtime, 'reference_network_check': reference,
              'sensor_limitations': 'ideal simulated own transforms/odometry/IMU; nominal intrinsics; no hardware test; lower ground-edge range differs from body center',
              'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}}
    (args.output / 'pixel_probe.json').write_bytes((json.dumps(report, indent=2, allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'runtime', 'reference_network_check')}, indent=2))


if __name__ == '__main__':
    main()
