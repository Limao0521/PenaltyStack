"""Preserve striker_v2's actor and fine-tune against visual balanced plus FSM."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from penalty.motion_contract import ROOT
from penalty.rl.striker_retraining_v2 import prepare_run, train, select, blind_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-policy', type=Path, default=ROOT/'models/rl/striker_v2')
    parser.add_argument('--motion-model', type=Path, default=ROOT/'models/vision/keeper_motion_v1.json')
    parser.add_argument('--stage', choices=('all', 'prepare', 'train', 'validation', 'select', 'test'), default='all')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--n-steps', type=int, default=128, help='steps per worker per PPO update')
    parser.add_argument('--updates', type=int, default=96)
    parser.add_argument('--checkpoint-every', type=int, default=32, help='validate every k completed PPO updates after training')
    parser.add_argument('--validation-episodes', type=int, default=64)
    parser.add_argument('--test-episodes', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--max-seconds', type=float, default=7200, help='training wall-clock guard; checks between updates')
    parser.add_argument('--retention-max-drop', type=float, default=.10, help='observed maximum FSM validation goal-rate drop')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest = prepare_run(args.output, base_policy=args.base_policy, motion_model=args.motion_model,
                           workers=args.workers, n_steps=args.n_steps, updates=args.updates,
                           checkpoint_every=args.checkpoint_every, validation_episodes=args.validation_episodes,
                           test_episodes=args.test_episodes, learning_rate=args.learning_rate,
                           max_seconds=args.max_seconds, resume=args.resume,
                           retention_max_drop=args.retention_max_drop)
    print('frozen run: %s (%s)' % (args.output, manifest['fingerprint']), flush=True)
    if args.stage in ('all', 'train'):
        train(args.output, manifest, resume_failed=args.resume)
    if args.stage in ('all', 'select', 'validation'):
        selected = select(args.output, manifest)
        print('validation selected update %d: %s' % (selected['update'], selected['validation_summary']), flush=True)
    if args.stage in ('all', 'test'):
        report = blind_test(args.output, manifest)
        for mode, summaries in report['summaries'].items():
            print('blind %s: %s, paired %s' % (mode, summaries, report['comparisons'][mode]), flush=True)


if __name__ == '__main__':
    main()
