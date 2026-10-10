"""Offline reference joint poses from original B-Human physics/motions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from penalty.motion_contract import MODEL_JOINT_NAMES
from penalty.rl.contracts import KeeperAction, StrikerAction
from penalty.rl.keeper_benchmark import VariedStrikerPolicy
from penalty.rl.observation import torso_roll_pitch
from penalty.sim_vision import CameraDuelWorld


def collect(seed=67001):
    world = CameraDuelWorld()
    poses = []
    try:
        for index, style in enumerate(('center', 'corners', 'soft', 'feint')):
            obs, _, _ = world.reset(np.random.default_rng(seed+index), {})
            policy = VariedStrikerPolicy(style, seed+index)
            for step in range(90):
                if step in (0, 12, 25, 40, 55, 70):
                    sim = world.sim
                    frame = sim.adapter.frame('striker')
                    roll, pitch = torso_roll_pitch(sim.data.xmat[sim.model.body('striker_Nao').id])
                    poses.append({'joints': {name: frame['angles'][i] for i, name in enumerate(MODEL_JOINT_NAMES)
                                             if not name.endswith('Hand')},
                                  'torso_roll': roll, 'torso_pitch': pitch})
                obs, _, result, _ = world.duel_step(StrikerAction.from_normalized(policy(obs.to_array())),
                                                   KeeperAction(0, 0., .5))
                if result is not None:
                    break
            print('%s: %d poses acumuladas' % (style, len(poses)), flush=True)
    finally:
        world.close()
    return {'version': 'bhuman_reference_articulations_v1', 'seed': seed,
            'purpose': 'offline visual reference generation only; never runtime rival sensing',
            'styles': ['center', 'corners', 'soft', 'feint'], 'poses': poses}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=67001)
    args = parser.parse_args()
    report = collect(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
