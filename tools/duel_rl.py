"""Cross-play in B-Human: every striker against every keeper, same episode seeds.

  python tools/duel_rl.py --strikers scripted v2=runs/train/X/striker/latest \
      --keepers scripted v4=runs/train/Y/keeper/latest:gate --episodes 200 --workers 30

A trained policy is NAME=DIRECTORY; ':gate' turns on the keeper's
commit_after_shot safety filter (use it for keepers trained with it).
Baselines: scripted (both roles), geometric and center (keeper).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from penalty.motion_contract import ROOT  # noqa: E402
from penalty.rl.duel import CONDITIONS, duel_matrix, player_spec  # noqa: E402
from penalty.rl.run_metadata import run_metadata  # noqa: E402


def parse_players(values):
    return dict(player_spec(value) for value in values)


def table(result, strikers, keepers):
    lines = ['%-12s' % 'tirador \\ portero' + ''.join(k.rjust(22) for k in keepers)]
    for s in strikers:
        cells = []
        for k in keepers:
            rates = result['pairs']['%s|%s' % (s, k)]['rates']
            cells.append(('goles %3.0f%% atajadas %3.0f%%' % (100*rates['goal_rate'],
                                                             100*rates['save_rate'])).rjust(22))
        lines.append('%-12s' % s + ''.join(cells))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--strikers', nargs='+', required=True)
    parser.add_argument('--keepers', nargs='+', required=True)
    parser.add_argument('--episodes', type=int, default=200, help='episodios por pareja')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--conditions', nargs='+', choices=sorted(CONDITIONS), default=['realista'])
    parser.add_argument('--output', type=Path, default=ROOT/'runs'/'duel.json')
    args = parser.parse_args()
    strikers, keepers = parse_players(args.strikers), parse_players(args.keepers)
    report = {'metadata': run_metadata(strikers={k: v.get('path', v.get('name')) for k, v in strikers.items()},
                                       keepers={k: dict(policy=v.get('path', v.get('name')), gate=v.get('gate', False))
                                                for k, v in keepers.items()},
                                       episodes_per_pair=args.episodes, seed=args.seed),
              'conditions': {}}
    for condition in args.conditions:
        result = duel_matrix(strikers, keepers, args.episodes, args.seed, args.workers, condition)
        report['conditions'][condition] = result
        print('\n== %s (%d episodios por pareja, %.0f s)' % (condition, args.episodes, result['wall_s']))
        print(table(result, strikers, keepers), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n', encoding='utf-8')
    print('informe: %s' % args.output)


if __name__ == '__main__':
    main()
