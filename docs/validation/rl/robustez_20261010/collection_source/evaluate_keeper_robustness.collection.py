"""Paired B-Human keeper benchmark, including delayed shots and changed aim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from penalty.rl.duel import export_trained
from penalty.rl.keeper_benchmark import STYLES, benchmark, keeper_spec
from penalty.rl.robust_keeper import RobustKeeperSettings
from penalty.rl.run_metadata import run_metadata
from penalty.motion_contract import ROOT, executable_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keepers', nargs='+', default=['robust', 'anticipation', 'center:gate'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=62000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--styles', nargs='+', choices=STYLES, default=STYLES)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--sweep', action='store_true', help='selección: tiempos y cobertura, sin tocar el test final')
    parser.add_argument('--condition', choices=('realista', 'stress'), default='realista')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = RobustKeeperSettings(**json.loads(args.settings.read_text(encoding='utf-8'))) if args.settings else RobustKeeperSettings()
    keepers = {name: keeper_spec(name, settings) for name in args.keepers}
    if args.sweep:
        for lead in (.7, 1.2, 1.8):
            for edge in (.3, .4):
                keepers['trajectory_%g_%g' % (lead, edge)] = keeper_spec('robust', RobustKeeperSettings(dive_lead_s=lead, corner_y=edge))
    trained = {s: export_trained('models/rl/striker_'+s) for s in ('v1', 'v2') if s in args.styles}
    metadata = run_metadata(selection=args.sweep, workers=args.workers,
                            benchmark_inputs=['robot/ball scene poses', 'synthetic perception uncertainty'])
    for name in ('penalty/rl/robust_keeper.py', 'penalty/rl/keeper_benchmark.py', 'penalty/rl/keeper_noise.py',
                 'penalty/rl/anticipation.py', 'penalty/rl/duel.py', 'models/rl/kick_side_predictor.json',
                 'tools/evaluate_keeper_robustness.py'):
        metadata['hashes'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    metadata['native_executable_sha256'] = hashlib.sha256(executable_path().read_bytes()).hexdigest()
    report = benchmark(keepers, trained, args.episodes, args.seed, args.workers, args.styles, args.condition)
    report['metadata'] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name, summary in report['summary'].items():
        print('%s: promedio %.3f | peor estilo %s %.3f | sim_error %d' % (
            name, summary['macro_save_rate'], summary['worst_style'], summary['worst_style_save_rate'], summary['sim_errors']))
    print('informe:', args.output)


if __name__ == '__main__':
    main()
