"""Evaluate the pre-fixed blind selector test (indices 1..20 for eight styles).

Reuses original model-bound test caches for indices 1..19 and adds index 20.
The model is frozen; integration-smoke index zero is excluded for all styles.
No training, network access or physical robot movements occur here.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.rl.selector_heldout import evaluate_heldout, initialize_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('runs/keeper_selector/v1'))
    parser.add_argument('--model', type=Path, default=Path('models/vision/keeper_selector_v1.json'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--init-protocol', action='store_true',
                        help='fix the protocol before fitting, then exit without running motors')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be in [1,16]')
    try:
        if args.init_protocol:
            protocol = initialize_protocol(args.output)
            print('Blind protocol fixed before fitting: '+str((args.output/'heldout_protocol.json').resolve()))
            print('%d cases: indices %d..%d; excluded=%s' %
                  (len(protocol['styles'])*protocol['episodes_per_style'],
                   protocol['episode_indices'][0], protocol['episode_indices'][-1],
                   protocol['excluded_episode_indices']))
            return 0
        report = evaluate_heldout(args.output, args.model, args.workers)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(2, str(error)+'\n')
    for mode, summary in report['summary'].items():
        saves = sum(style['counts']['save'] for style in summary['per_style'].values())
        total = sum(style['episodes'] for style in summary['per_style'].values())
        print('%s: %d/%d saves; sim_error=%d; no_commit=%d; defenses=%s' %
              (mode, saves, total, summary['sim_errors'], report['no_commit_counts'][mode],
               report['defense_counts'][mode]))
    print('Blind final test: '+str((args.output/'heldout_report.json').resolve()))
    return 1 if any(report['sim_error_counts'].values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
