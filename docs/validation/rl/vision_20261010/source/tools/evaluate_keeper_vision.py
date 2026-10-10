"""Evaluate vision keeper vs project FSM using rendered NAO-camera RGB."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from penalty.rl.keeper_benchmark import STYLES
    from penalty.rl.vision_benchmark import benchmark
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=5, help='per style and keeper')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=65001)
    parser.add_argument('--styles', nargs='+', choices=STYLES, default=STYLES)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.episodes, args.seed, args.workers, tuple(args.styles))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Reporte: %s' % args.output.resolve())


if __name__ == '__main__':
    main()
