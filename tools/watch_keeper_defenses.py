"""Run one camera/sensor penalty with a named-defense keeper; optional live viewer."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.keeper_defenses import catalog
from penalty.keeper_library import MODES, VisionLibraryKeeper
from penalty.rl.defense_benchmark import SensorDefenseDuelWorld, camera_episode
from penalty.rl.duel import make_player, player_spec
from penalty.rl.keeper_benchmark import STYLES, VariedStrikerPolicy
from penalty.sim_orientation import SensorBHumanKeeperCamera


class LiveDefenseWorld(SensorDefenseDuelWorld):
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
            self.sim.mutation_lock = self.viewer.lock
            self.wall, self.sim_origin = time.monotonic(), float(self.sim.data.time)
        if not self.viewer.is_running():
            raise RuntimeError('visor cerrado por el usuario')
        ahead = float(self.sim.data.time)-self.sim_origin-(time.monotonic()-self.wall)
        if ahead > 0:
            time.sleep(min(ahead, .02))
        # Native pipes and camera inference execute outside viewer.lock().
        self.viewer.sync()

    def close(self):
        try:
            if self.session is not None:
                self.sim.mutation_lock = nullcontext
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        finally:
            super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, default='balanced')
    parser.add_argument('--style', choices=STYLES, default='v2')
    parser.add_argument('--seed', type=int, default=72001000000)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/keeper_defenses/watch'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    world = (LiveDefenseWorld if args.live else SensorDefenseDuelWorld)(
        ball_jitter=.05, start_jitter=(.05, .1))
    camera = None
    trace_path = args.output/'trace.jsonl'
    with trace_path.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps({'type': 'METADATA', 'policy': VisionLibraryKeeper(args.mode).describe(),
                                 'catalog': catalog(), 'seed': args.seed, 'style': args.style,
                                 'simulation_only': True, 'ideal_sensor_readings': True,
                                 'inference_advances_physics': False}, allow_nan=False)+'\n')

        def frame(state, action, decision):
            # Perception state only; this trace contains no evaluation truth.
            stream.write(json.dumps({'type': 'FRAME', 'vision': state,
                                     'action': action.to_dict(), 'decision': decision},
                                    allow_nan=False)+'\n')

        try:
            world._ensure_session()
            camera = SensorBHumanKeeperCamera(world)
            striker = (make_player(player_spec('models/rl/striker_'+args.style)[1], 'striker')
                       if args.style in ('v1', 'v2') else VariedStrikerPolicy(args.style, args.seed))
            result = camera_episode(world, camera, striker, args.mode, args.seed, on_frame=frame)
        finally:
            try:
                if camera is not None:
                    camera.close()
            finally:
                world.close()
    (args.output/'episode.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'result': result['result'], 'mode': args.mode,
                      'defense': result['commit_defense'], 'decision': result['keeper_decision'],
                      'trace': str(trace_path.resolve())}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
