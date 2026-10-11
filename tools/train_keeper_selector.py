"""Train/evaluate six original defenses with the unchanged vision keeper v2 gate.

Example (real B-Human/MuJoCo physics, RGB cameras and simulated own sensors):
  python tools/train_keeper_selector.py --train-episodes 12 --validation-episodes 6 \
      --test-episodes 20 --workers 4 --output runs/keeper_selector/v1
  python tools/train_keeper_selector.py --output runs/keeper_selector/v1 --resume

Every collected case executes all six defenses. The final test compares a
frozen learned selector, original v2 and balanced using disjoint seeds.
This trains tactical selection only; no joint trajectories or hardware moves.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.rl.keeper_benchmark import STYLES
from penalty.rl.selector_training import run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-episodes', type=int, default=12, help='paired contexts per style; six duels each')
    parser.add_argument('--validation-episodes', type=int, default=6, help='selection contexts per style; six duels each')
    parser.add_argument('--test-episodes', type=int, default=20, help='held-out cases per style; three policies each')
    parser.add_argument('--styles', nargs='+', choices=STYLES, default=list(STYLES))
    parser.add_argument('--workers', type=int, default=4, help='independent RGB worlds, in [1,16]')
    parser.add_argument('--fit-epochs', type=int, default=1500)
    parser.add_argument('--output', type=Path, default=Path('runs/keeper_selector/v1'))
    parser.add_argument('--model-output', type=Path, default=Path('models/vision/keeper_selector_v1.json'))
    parser.add_argument('--stage', choices=('collect', 'fit', 'evaluate', 'all'), default='all')
    parser.add_argument('--resume', action='store_true', help='reuse only matching, complete per-case checkpoints')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be in [1,16]')
    try:
        result = run(args.output, model_output=args.model_output, stage=args.stage,
                     train_episodes=args.train_episodes, validation_episodes=args.validation_episodes,
                     test_episodes=args.test_episodes, styles=args.styles, workers=args.workers,
                     fit_epochs=args.fit_epochs, resume=args.resume)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(2, str(error)+'\n')
    errors = 0
    for split, dataset in result.get('datasets', {}).items():
        summary = dataset['summary']
        errors += summary['sim_errors']+summary['pairing_mismatches']
        print('%s: %d/%d eligible contexts; %d duels; sim_error=%d; pairing_mismatch=%d' %
              (split, summary['eligible_contexts'], summary['contexts'], summary['duels'],
               summary['sim_errors'], summary['pairing_mismatches']))
    if 'test' in result:
        report = result['test']
        for mode, summary in report['summary'].items():
            count = sum(style['counts']['save'] for style in summary['per_style'].values())
            total = sum(style['episodes'] for style in summary['per_style'].values())
            errors += summary['sim_errors']
            print('%s: %d/%d saves; sim_error=%d; no_commit=%d; defenses=%s' %
                  (mode, count, total, summary['sim_errors'], report['no_commit_counts'][mode],
                   report['defense_counts'][mode]))
        print('Frozen final test: '+str((args.output/'test_report.json').resolve()))
    print('Run: '+str(args.output.resolve()))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
