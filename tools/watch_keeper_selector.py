"""Watch one sensor-camera penalty using the trained six-defense selector."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.keeper_defenses import catalog
from penalty.keeper_selector import LearnedDefenseKeeper, load_model
from penalty.rl.defense_benchmark import SensorDefenseDuelWorld
from penalty.rl.duel import make_player, player_spec
from penalty.rl.keeper_benchmark import STYLES, VariedStrikerPolicy
from penalty.rl.selector_training import run_camera_episode
from penalty.sim_orientation import SensorBHumanKeeperCamera
from tools.watch_keeper_defenses import LiveDefenseWorld


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
                        default=ROOT/'models/vision/keeper_selector_v1.json')
    parser.add_argument('--style', choices=STYLES, default='mixed')
    parser.add_argument('--seed', type=int, default=77001000000)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/keeper_selector/watch'))
    args = parser.parse_args()
    keeper = LearnedDefenseKeeper(load_model(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    world = (LiveDefenseWorld if args.live else SensorDefenseDuelWorld)(
        ball_jitter=.05, start_jitter=(.05, .1))
    camera = None
    trace_path = args.output/'trace.jsonl'
    with trace_path.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps({
            'type': 'METADATA', 'policy': keeper.describe(), 'catalog': catalog(),
            'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'seed': args.seed, 'style': args.style, 'simulation_only': True,
            'ideal_sensor_readings': True, 'inference_advances_physics': False,
        }, allow_nan=False)+'\n')

        def frame(state, action, decision):
            stream.write(json.dumps({'type': 'FRAME', 'vision': state,
                                     'action': action.to_dict(), 'decision': decision},
                                    allow_nan=False)+'\n')

        try:
            world._ensure_session()
            camera = SensorBHumanKeeperCamera(world)
            striker = (make_player(player_spec('models/rl/striker_'+args.style)[1], 'striker')
                       if args.style in ('v1', 'v2')
                       else VariedStrikerPolicy(args.style, args.seed))
            result = run_camera_episode(world, camera, striker, keeper, args.seed,
                                        on_frame=frame)
        finally:
            try:
                if camera is not None:
                    camera.close()
            finally:
                world.close()
    (args.output/'episode.json').write_text(
        json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'result': result['result'], 'defense': result.get('commit_defense'),
                      'decision': result.get('keeper_decision'),
                      'trace': str(trace_path.resolve())}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
