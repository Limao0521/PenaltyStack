"""Search runtime-supported kick parameters against a fixed tactical target.

Each candidate is scored with kick_objective_v1 over `episodes` penalties
per seed. Only ParameterizedKick.bounds() is searched; see
penalty/skills/kick.py for what the B-Human runtime exposes today.

  python -m penalty.optimization.kick_search --method random --budget 8
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from penalty.motion_contract import ROOT
from penalty.optimization.objective import KICK_OBJECTIVE_V1, kick_objective
from penalty.optimization.search import cma_es, random_search
from penalty.rl.contracts import KICK_LEFT, KICK_RIGHT, StrikerAction
from penalty.rl.evaluation import make_env, rollout
from penalty.rl.run_metadata import run_metadata
from penalty.skills.kick import ParameterizedKick


class KickEvaluator:
    def __init__(self, backend='bhuman', opponent='static', target_y=.4, kick_side=KICK_LEFT,
                 seeds=(0, 1, 2), episodes=1):
        self.kick = ParameterizedKick(kick_side)
        self.names = tuple(self.kick.bounds())
        self.target_y, self.kick_side = target_y, kick_side
        self.seeds, self.episodes = tuple(seeds), episodes
        self.env = make_env('striker', backend, opponent)
        self.log = []

    def bounds(self):
        return [self.kick.bounds()[name] for name in self.names]

    def __call__(self, vector):
        params = self.kick.validate(ParameterizedKick.from_vector(vector, self.names))
        action = StrikerAction(self.target_y, self.kick_side, params.power, 1.).to_normalized()
        records = [rollout(self.env, lambda _observation: action, seed*1000 + i)
                   for seed in self.seeds for i in range(self.episodes)]
        value = kick_objective(records)
        self.log.append({'parameters': asdict(params), 'objective': value, 'records': records})
        return value

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=('random', 'cmaes'), default='random')
    parser.add_argument('--budget', type=int, default=12)
    parser.add_argument('--backend', choices=('synthetic', 'bhuman'), default='bhuman')
    parser.add_argument('--opponent', default='static')
    parser.add_argument('--target-y', type=float, default=.4)
    parser.add_argument('--kick-side', choices=('left', 'right'), default='left')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--episodes', type=int, default=1, help='episodios por semilla y candidato')
    parser.add_argument('--output', type=Path, default=ROOT/'runs'/'best_kick_params.json')
    args = parser.parse_args()
    side = KICK_LEFT if args.kick_side == 'left' else KICK_RIGHT
    evaluator = KickEvaluator(args.backend, args.opponent, args.target_y, side, args.seeds,
                              args.episodes)
    try:
        search = random_search if args.method == 'random' else cma_es
        result = search(evaluator, evaluator.bounds(), args.budget, seed=args.seeds[0])
    finally:
        evaluator.close()
    best = max(evaluator.log, key=lambda item: item['objective'])
    report = {'metadata': run_metadata(method=args.method, backend=args.backend,
                                       opponent=args.opponent, target_y=args.target_y,
                                       kick_side=args.kick_side, seeds=args.seeds,
                                       episodes=args.episodes, objective=KICK_OBJECTIVE_V1),
              'searched': list(evaluator.names), 'best_parameters': best['parameters'],
              'best_objective': best['objective'], 'evaluations': result['evaluations'],
              'candidates': evaluator.log}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'best_parameters': best['parameters'], 'best_objective': best['objective']}))


if __name__ == '__main__':
    main()
