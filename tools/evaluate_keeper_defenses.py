"""Compare original B-Human keeper guards with unchanged v2 camera prediction."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.keeper_library import MODES
from penalty.rl.defense_benchmark import benchmark
from penalty.rl.keeper_benchmark import STYLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--styles', nargs='+', choices=STYLES, default=list(STYLES))
    parser.add_argument('--episodes', type=int, default=2, help='episodes per mode and style')
    parser.add_argument('--seed', type=int, default=72001)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('runs/keeper_defenses/smoke_72001.json'))
    args = parser.parse_args()
    report = benchmark(args.modes, args.episodes, args.seed, args.workers, args.styles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for mode, group in report['records'].items():
        rows = [row for records in group.values() for row in records]
        print('%s: %d/%d saves; sim_error=%d; defenses=%s' %
              (mode, sum(row['result'] == 'save' for row in rows), len(rows),
               sum(row['result'] == 'sim_error' for row in rows), report['defense_counts'][mode]))
    print('Report: '+str(args.output.resolve()))
    return 1 if any(row['result'] == 'sim_error' for group in report['records'].values()
                    for rows in group.values() for row in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
