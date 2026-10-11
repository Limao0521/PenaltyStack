"""Train an actual bounded B-Human keeper motion skill, simulation only."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.rl.motion_training import (
    prepare_run, safety_screen, search, validate_and_fit, blind_test,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/keeper_motion/v1')
    parser.add_argument('--model-output', type=Path, default=ROOT/'models/vision/keeper_motion_v1.json')
    parser.add_argument('--stage', choices=('all', 'safety', 'search', 'fit', 'test'), default='all')
    parser.add_argument('--candidates', type=int, default=16)
    parser.add_argument('--train-episodes', type=int, default=4)
    parser.add_argument('--validation-episodes', type=int, default=8)
    parser.add_argument('--test-episodes', type=int, default=20)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest = prepare_run(args.output, args.candidates, args.train_episodes,
                           args.validation_episodes, args.test_episodes, args.resume)
    if args.stage in ('all', 'safety'):
        result = safety_screen(args.output, manifest, args.workers)
        print('Safe profiles: '+', '.join(result['eligible_profiles']), flush=True)
    if args.stage in ('all', 'search'):
        result = search(args.output, manifest, args.workers)
        print('Train shortlist: '+', '.join(profile['id'] for profile in result['shortlist']), flush=True)
    if args.stage in ('all', 'fit'):
        result = validate_and_fit(args.output, manifest, args.model_output, args.workers)
        print('Frozen selected profile: '+str(result['selected_profile']), flush=True)
    if args.stage in ('all', 'test'):
        result = blind_test(args.output, manifest, args.model_output, args.workers)
        print('Blind test saves: '+str(result['save_counts']), flush=True)
        print('Paired result: '+str(result['comparisons']['trained_vs_balanced']), flush=True)


if __name__ == '__main__':
    main()
