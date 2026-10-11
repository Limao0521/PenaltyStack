"""Measure real striker_v2 saturation before the fixed PPO fine-tuning run."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from penalty.rl.striker_calibration import calibrate_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('models/rl/striker_v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    calibrate_policy(args.base, args.output, args.episodes, args.workers)


if __name__ == '__main__':
    main()
