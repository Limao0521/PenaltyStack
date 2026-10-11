"""Paired keeper duels on the camera/own-sensor pipeline: FSM, v2, v3, v4, v5.

Example (final test, new seeds, every keeper sees the same kicks):
  python tools/evaluate_keeper_fusion.py --keepers scripted vision pose pose_v4 ball_v5 full_v5 \
      --episodes 50 --seed 70001 --workers 16 --output runs/keeper_fusion/final.json
`--frames` also writes compact per-frame traces (JSONL) next to the report.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from penalty.rl.fusion_benchmark import KEEPERS, benchmark
    from penalty.rl.keeper_benchmark import STYLES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--keepers', nargs='+', choices=tuple(KEEPERS), required=True)
    parser.add_argument('--episodes', type=int, default=50, help='per style and keeper')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)//2))
    parser.add_argument('--seed', type=int, default=70001)
    parser.add_argument('--styles', nargs='+', choices=STYLES, default=STYLES)
    parser.add_argument('--frames', action='store_true', help='store compact per-frame traces')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log', type=Path, help='also append progress lines to this file')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames_path = args.output.with_suffix('.frames.jsonl') if args.frames else None
    log = open(args.log, 'a', encoding='utf-8') if args.log else None

    def progress(line, flush=True):
        print(line, flush=True)
        if log is not None:
            log.write(line+'\n')
            log.flush()
    try:
        progress('Porteros %s; %d episodios por estilo; semilla %d; %d workers' % (
            ' '.join(args.keepers), args.episodes, args.seed, args.workers))
        report = benchmark(tuple(args.keepers), args.episodes, args.seed, args.workers, tuple(args.styles),
                           frames=args.frames, frames_path=frames_path, progress=progress)
        args.output.write_text(json.dumps(report, indent=1, allow_nan=False)+'\n', encoding='utf-8')
        for name, summary in report['summary'].items():
            progress('%-9s media entre estilos %.1f %%, peor %s %.0f %%, errores %d, compromisos tempranos %d' % (
                name, 100*summary['macro_save_rate'], summary['worst_style'],
                100*summary['worst_style_save_rate'], summary['sim_errors'], summary['early_commits']))
        for name, comparison in report['comparisons'].items():
            progress('%s: ganancia %+.1f pp (IC95 %+.1f a %+.1f)' % (
                name, 100*comparison['macro_gain'], *(100*v for v in comparison['macro_gain_ci95'])))
        progress('Reporte: %s' % args.output.resolve())
    except BaseException:
        import traceback
        progress('ERROR:\n' + traceback.format_exc())
        raise
    finally:
        if log is not None:
            log.close()

if __name__ == '__main__':
    main()
