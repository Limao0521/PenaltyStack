"""Run a frozen motor-skill profile on one RGB/sensor penalty in MuJoCo."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.keeper_library import VisionLibraryKeeper
from penalty.rl.motion_training import load_profile
from penalty.rl.motion_world import MotionDuelWorld
from penalty.rl.keeper_benchmark import STYLES
from penalty.rl.duel import player_spec
from penalty.rl.selector_training import _striker, _close_resources, run_camera_episode
from penalty.skills.keeper_motion import MotionParameters
from penalty.sim_orientation import SensorBHumanKeeperCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'models/vision/keeper_motion_v1.json')
    parser.add_argument('--style', choices=STYLES, default='mixed')
    parser.add_argument('--seed', type=int, default=84001000000)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/keeper_motion/watch')
    args = parser.parse_args()
    model = load_profile(args.model)
    params = MotionParameters(**model['parameters'])
    args.output.mkdir(parents=True, exist_ok=True)
    world = MotionDuelWorld(params, ball_jitter=.05, start_jitter=(.05, .1))
    camera = None
    with (args.output/'trace.jsonl').open('w', encoding='utf-8') as stream:
        stream.write(json.dumps({'type': 'METADATA', 'model': model,
                                 'parameters': asdict(params), 'style': args.style,
                                 'seed': args.seed, 'hardware_validated': False}, allow_nan=False)+'\n')
        def frame(state, action, decision):
            stream.write(json.dumps({'type': 'FRAME', 'vision': state,
                                     'action': action.to_dict(), 'decision': decision}, allow_nan=False)+'\n')
        try:
            world._ensure_session()
            camera = SensorBHumanKeeperCamera(world)
            trained = {args.style: player_spec('models/rl/striker_'+args.style)[1]} if args.style in ('v1', 'v2') else {}
            result = run_camera_episode(world, camera, _striker(args.style, args.seed, trained),
                                        VisionLibraryKeeper('balanced'), args.seed, on_frame=frame)
        finally:
            _close_resources(world, camera)
    (args.output/'episode.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'result': result['result'], 'defense': result['commit_defense'],
                      'profile': params.__dict__, 'output': str(args.output.resolve())}, indent=2))


if __name__ == '__main__':
    main()
