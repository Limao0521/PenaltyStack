"""Overnight PPO training of the tactical striker and keeper.

  python tools/train_rl.py --hours 10
  python tools/train_rl.py --output runs/train/NOMBRE --resume   # continuar tras un corte

Synthetic curriculum first (minutes), then B-Human/MuJoCo stages until the
wall-clock budget ends. Results: <output>/report.json, per-role policies in
<output>/<role>/latest and <output>/<role>/<stage>, CSV logs in <output>/logs.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from penalty.motion_contract import ROOT  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hours', type=float, default=10., help='presupuesto total de reloj')
    parser.add_argument('--roles', nargs='+', choices=('striker', 'keeper'),
                        default=['striker', 'keeper'])
    parser.add_argument('--workers', type=int, default=8, help='entornos B-Human en paralelo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='prueba completa corta (usa --hours 0.25)')
    args = parser.parse_args()
    if not 0 < args.hours <= 72 or not 1 <= args.workers <= 16:
        parser.error('hours debe estar en (0, 72] y workers en [1, 16]')
    output = args.output or ROOT/'runs'/'train'/datetime.now().strftime('%Y%m%d_%H%M')
    if args.resume and not (output/'progress.json').is_file():
        parser.error('--resume necesita --output con una ejecución existente')
    from penalty.rl.training import train
    print('salida: %s' % output, flush=True)
    try:
        train(output, hours=args.hours, roles=tuple(args.roles), workers=args.workers,
              seed=args.seed, resume=args.resume, smoke=args.smoke)
    except ValueError as error:
        parser.exit(1, str(error) + '\n')
    print('terminado: %s' % (output/'report.json'), flush=True)


if __name__ == '__main__':
    main()
