"""Run/replay one sensor-camera penalty; optional live MuJoCo viewer.

The keeper receives rendered RGB and its own sensors. Ground truth in saved
evaluation_only rows is used for validation, never fed back to perception.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.keeper_pose_vision import VisionPoseKeeper
from penalty.rl.duel import make_player, player_spec
from penalty.rl.keeper_benchmark import STYLES, VariedStrikerPolicy
from penalty.rl.orientation_benchmark import camera_episode, perception_summary
from penalty.sim_orientation import SensorCameraDuelWorld, SensorBHumanKeeperCamera


class LiveSensorWorld(SensorCameraDuelWorld):
    def __init__(self, **options):
        super().__init__(**options)
        self.viewer = None
        self.wall = self.sim_origin = None

    def _tick(self):
        super()._tick()
        if self.session.state != 'RUNNING':
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
            self.viewer.cam.lookat[:] = (-.6, 0., .2)
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 2.8, 180, -25
            self.wall, self.sim_origin = time.monotonic(), float(self.sim.data.time)
        if not self.viewer.is_running():
            raise RuntimeError('visor cerrado por el usuario')
        ahead = float(self.sim.data.time)-self.sim_origin-(time.monotonic()-self.wall)
        if ahead > 0:
            time.sleep(min(ahead, .02))
        # Never hold viewer.lock() during camera/native inference.
        self.viewer.sync()

    def close(self):
        try:
            if self.viewer:
                self.viewer.close()
        finally:
            super().close()


def replay(path, output):
    policy = VisionPoseKeeper()
    rows = []
    last_seed = last_time = None
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        state = row['vision']
        if row.get('seed') != last_seed or (last_time is not None and state['time_s'] < last_time):
            policy.reset()
        last_seed, last_time = row.get('seed'), state['time_s']
        rows.append({'vision': state, 'action': policy(state), 'decision': dict(policy.decision),
                     'executed': False})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(('\n'.join(json.dumps(row, allow_nan=False) for row in rows)+'\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', choices=STYLES, default='v2')
    parser.add_argument('--seed', type=int, default=68003)
    parser.add_argument('--keeper', choices=('pose', 'vision', 'scripted'), default='pose')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--replay', type=Path, help='offline trace replay; no simulation, viewer or actuators')
    args = parser.parse_args()
    if args.replay:
        if args.live:
            parser.error('--replay and --live are incompatible')
        replay(args.replay, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    world = (LiveSensorWorld if args.live else SensorCameraDuelWorld)(ball_jitter=.05, start_jitter=(.05, .1))
    camera = None
    trace = []
    def on_frame(world, camera, state, action, decision):
        if state['opponent']['yaw_valid'] and not (args.output/'camera.png').exists():
            from PIL import Image
            Image.fromarray(camera.rgb).save(args.output/'camera.png')
    try:
        world._ensure_session()
        camera = SensorBHumanKeeperCamera(world)
        striker = (make_player(player_spec('models/rl/striker_'+args.style)[1], 'striker') if args.style in ('v1', 'v2')
                   else VariedStrikerPolicy(args.style, args.seed))
        result = camera_episode(world, camera, striker, args.keeper, args.seed, trace=trace, on_frame=on_frame)
    finally:
        try:
            if camera:
                camera.close()
        finally:
            world.close()
    (args.output/'trace.jsonl').write_bytes(('\n'.join(json.dumps(row, allow_nan=False) for row in trace)+'\n').encode('utf-8'))
    (args.output/'episode.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'result': result['result'], 'seed': args.seed, 'keeper': args.keeper,
                     'perception': perception_summary(result['opponent_measurements']),
                     'trace': str((args.output/'trace.jsonl').resolve())}, indent=2))


if __name__ == '__main__':
    main()
