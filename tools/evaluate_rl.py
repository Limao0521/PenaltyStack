"""Benchmark a tactical policy and write a JSON report with provenance.

Examples:
  python tools/evaluate_rl.py --role keeper --policy geometric --episodes 100 --seeds 0 1 2 3 4
  python tools/evaluate_rl.py --role striker --backend bhuman --episodes 5 --workers 4
  python tools/evaluate_rl.py --role striker --throughput 1 4 8 12 --backend bhuman --episodes 4
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from penalty.motion_contract import ROOT  # noqa: E402
from penalty.rl.evaluation import evaluate  # noqa: E402
from penalty.rl.run_metadata import run_metadata  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--role', choices=('striker', 'keeper'), required=True)
    parser.add_argument('--policy', default='scripted', help='scripted | geometric (keeper) | random')
    parser.add_argument('--backend', choices=('synthetic', 'bhuman'), default='synthetic')
    parser.add_argument('--opponent', default=None)
    parser.add_argument('--episodes', type=int, default=100, help='episodios por semilla')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--throughput', type=int, nargs='*', default=None,
                        help='medir con estas cantidades de procesos (p. ej. 1 4 8 12)')
    parser.add_argument('--records', action='store_true', help='incluir cada episodio en el informe')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1:
        parser.error('episodes y workers deben ser >= 1')
    report = {'metadata': run_metadata(role=args.role, policy=args.policy, backend=args.backend,
                                       opponent=args.opponent, seeds=args.seeds,
                                       episodes_per_seed=args.episodes, model=None)}
    if args.throughput:
        report['throughput'] = []
        for workers in args.throughput:
            result = evaluate(args.role, args.policy, args.backend, args.opponent,
                              args.episodes, args.seeds[:1], workers)
            report['throughput'].append(result['throughput'])
            print(json.dumps(result['throughput']), flush=True)
    else:
        result = evaluate(args.role, args.policy, args.backend, args.opponent,
                          args.episodes, args.seeds, args.workers)
        if not args.records:
            result.pop('records')
        report.update(result)
        print(json.dumps(result['summary']['rates'], indent=2))
    output = args.output or ROOT/'runs'/('eval_%s_%s_%s.json' % (args.role, args.policy, args.backend))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('informe: %s' % output)


if __name__ == '__main__':
    main()
