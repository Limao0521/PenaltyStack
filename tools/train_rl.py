"""PPO training of the tactical striker and keeper until convergence.

  python tools/train_rl.py --hours 72 --workers 16
  python tools/train_rl.py --output runs/train/NOMBRE --resume   # continuar tras un corte

Synthetic curriculum first (minutes), then B-Human/MuJoCo stages that stop
when their outcome rate converges, with --hours as the overall cap. The run
is supervised: a crash or hang restarts it from the last checkpoint; an
unhealthy simulator stops it for good. Results: <output>/report.json,
per-role policies in <output>/<role>/latest, <output>/<role>/<stage> and
<output>/<role>/<stage>_best, CSV logs in <output>/logs.
"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from penalty.motion_contract import ROOT  # noqa: E402

EXIT_ABORTED = 3   # penalty.rl.training.EXIT_ABORTED, kept import-free for the supervisor
HEALTHY_RUN_S = 1800
RESTART_DELAY_S = 30


def supervise(run, output, max_restarts, resume=False, sleep=time.sleep, clock=time.monotonic):
    """run(resume) -> exit code. Restarts unexpected failures from the last
    checkpoint; a run that lasted HEALTHY_RUN_S resets the failure count."""
    failures = 0
    while True:
        started = clock()
        code = run(resume)
        if code == 0:
            return 0
        if code == EXIT_ABORTED or not (Path(output)/'progress.json').is_file():
            return code
        failures = 0 if clock() - started >= HEALTHY_RUN_S else failures + 1
        if failures > max_restarts:
            print('%d fallos seguidos; se detiene' % failures, flush=True)
            return code
        print('el entrenamiento terminó con código %d (fallos seguidos %d/%d); se reanuda desde el '
              'último checkpoint en %d s' % (code, failures, max_restarts, RESTART_DELAY_S), flush=True)
        sleep(RESTART_DELAY_S)
        resume = True


def _child_command(args, output, resume):
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--output', str(output),
               '--hours', str(args.hours), '--workers', str(args.workers), '--seed', str(args.seed),
               '--roles', *args.roles]
    return command + ['--smoke'] * args.smoke + ['--resume'] * resume


def _run_child(args, output, resume):
    process = subprocess.Popen(_child_command(args, output, resume))
    try:
        return process.wait()
    except KeyboardInterrupt:
        print('interrumpido; esperando el checkpoint final...', flush=True)
        process.wait()
        sys.exit(130)


def _train(args, output, parser):
    from penalty.rl.training import TrainingAborted, train
    try:
        train(output, hours=args.hours, roles=tuple(args.roles), workers=args.workers,
              seed=args.seed, resume=args.resume, smoke=args.smoke)
    except TrainingAborted as error:
        parser.exit(EXIT_ABORTED, 'abortado: %s\n' % error)
    except ValueError as error:  # configuration error: restarting would not help
        parser.exit(EXIT_ABORTED, str(error) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hours', type=float, default=10., help='tope total de reloj')
    parser.add_argument('--roles', nargs='+', choices=('striker', 'keeper'),
                        default=['striker', 'keeper'])
    parser.add_argument('--workers', type=int, default=8, help='entornos B-Human en paralelo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='prueba completa corta (usa --hours 0.25)')
    parser.add_argument('--max-restarts', type=int, default=3,
                        help='reinicios seguidos tras un fallo antes de rendirse')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.hours <= 72 or not 1 <= args.workers <= 32 or args.max_restarts < 0:
        parser.error('hours debe estar en (0, 72], workers en [1, 32] y max-restarts >= 0')
    output = args.output or ROOT/'runs'/'train'/datetime.now().strftime('%Y%m%d_%H%M')
    if args.resume and not (output/'progress.json').is_file():
        parser.error('--resume necesita --output con una ejecución existente')
    if args.child:
        _train(args, output, parser)
        return
    print('salida: %s' % output, flush=True)
    code = supervise(lambda resume: _run_child(args, output, resume), output,
                     args.max_restarts, args.resume)
    if code:
        sys.exit(code)
    print('terminado: %s' % (output/'report.json'), flush=True)


if __name__ == '__main__':
    main()
