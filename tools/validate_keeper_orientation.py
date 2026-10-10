"""Held-out physical sensor/camera orientation probe at many initial yaws.

Only initial trial layouts use angles. Free physics and original B-Human
servos then run normally. Rival poses are read separately for evaluation.
"""
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

from penalty.rl.controllers import StaticKeeperController
from penalty.rl.orientation_benchmark import measurements, perception_summary
from penalty.sim_orientation import SensorCameraDuelWorld, SensorBHumanKeeperCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=16, help='per angular layout, .12 s intervals')
    parser.add_argument('--seed', type=int, default=68002)
    parser.add_argument('--angles', type=float, nargs='+', default=list(range(-177, 180, 30)))
    args = parser.parse_args()
    if args.frames < 3 or not all(math.isfinite(v) and -180 <= v <= 180 for v in args.angles):
        parser.error('at least 3 frames and finite angles in [-180,180] required')
    args.output.mkdir(parents=True, exist_ok=True)
    world, camera = SensorCameraDuelWorld(), None
    rows, layouts = [], []
    try:
        for index, angle in enumerate(args.angles):
            world._ensure_session()
            own_yaw = math.radians((-1)**index*8)
            # Initial trial offsets, not updates to an active root trajectory.
            world.sim.start_offsets = {'striker': (.015*(-1)**index, .035*math.sin(index), math.radians(angle)-math.pi),
                                       'keeper': (0., .015*(-1)**index, own_yaw)}
            world.reset(np.random.default_rng(args.seed+index), {})
            world.session.controllers['striker'] = StaticKeeperController()
            if camera is None:
                camera = SensorBHumanKeeperCamera(world)
            camera.reset()
            frames = []
            for frame in range(args.frames):
                state = camera.observe()
                measurement = measurements(world, state)
                frames.append(measurement)
                rows.append({'layout': index, 'initial_rival_yaw_deg': angle, 'vision': state,
                             'evaluation_only': measurement, 'executed_on_hardware': False})
                if frame == args.frames-1:
                    Image.fromarray(camera.rgb).save(args.output/('camera_%+04d.png' % angle))
                world._advance(until=lambda: world.session.penalty_time >= (frame+1)*.12)
            layouts.append({'initial_rival_yaw_deg': angle, **perception_summary(frames)})
            print('Yaw inicial %+g: %d/%d válidos' % (angle, sum(f['yaw_valid'] for f in frames), len(frames)), flush=True)
    finally:
        try:
            if camera:
                camera.close()
        finally:
            world.close()
    trace = args.output/'trace.jsonl'
    trace.write_bytes(('\n'.join(json.dumps(row, allow_nan=False) for row in rows)+'\n').encode('utf-8'))
    report = {'validation': 'keeper_sensor_orientation_probe_v1', 'seed': args.seed,
              'frames_per_layout': args.frames, 'layouts': layouts,
              'summary': perception_summary([row['evaluation_only'] for row in rows]),
              'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
              'sensor_limitations': 'ideal simulated encoder/IMU/FSR measurements; FK/attitude/contact odometry estimated; simulation appearance only; no hardware',
              'privileged_truth': 'evaluation_only fields never feed perception or keeper'}
    (args.output/'probe.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
