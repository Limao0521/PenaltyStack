"""Fit the keeper's kick-side predictor from B-Human penalties.

  python tools/fit_kick_predictor.py --episodes 600 --workers 30

The keeper stands still; for each penalty the striker's pose is taken from
the keeper's last observation before the opponent touches the ball, and the
label is where the ball crossed the goal line (or stopped). Writes
models/rl/kick_side_predictor.json with 5-fold cross-validated accuracy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from penalty.rl.anticipation import DEFAULT_PREDICTOR, KickSidePredictor, pose_features, side_class  # noqa: E402
from penalty.rl.training import CORNER_KICKS  # noqa: E402

WORLD = {k: v for k, v in CORNER_KICKS.items() if k != 'perception_noise'}


def collect(seeds):
    from penalty.rl.contracts import KeeperAction
    from penalty.rl.evaluation import make_env
    env = make_env('keeper', 'bhuman', 'random_kicks', dict(WORLD, seed=11))
    world, wait = env.unwrapped.world, KeeperAction(0, 0., .5).to_normalized()
    rows = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            before, done = None, False
            while not done:
                if world.contact_time is None:
                    before = observation
                observation, _, terminated, truncated, _ = env.step(wait)
                done = terminated or truncated
            if world.session is None or before is None or world.contact_time is None:
                continue  # sim_error or no kick
            crossing = world.sim.goal_crossing
            final_y = crossing[1] if crossing is not None else world._ball_xy()[1]
            rows.append((pose_features(before).tolist(), side_class(final_y)))
    finally:
        env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--seed', type=int, default=70000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, default=DEFAULT_PREDICTOR)
    args = parser.parse_args()
    seeds = list(range(args.seed, args.seed + args.episodes))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for chunk in pool.map(collect, [seeds[k::args.workers] for k in range(args.workers)])
                for row in chunk]
    x, y = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    folds, predicted = np.arange(len(y)) % 5, np.empty_like(y)
    for k in range(5):
        model = KickSidePredictor.fit(x[folds != k], y[folds != k])
        predicted[folds == k] = model.probabilities(x[folds == k]).argmax(1)
    corners = y != 1
    metadata = {'date': date.today().isoformat(), 'episodes': len(y), 'seeds': [seeds[0], seeds[-1]],
                'world': {k: list(v) if isinstance(v, tuple) else v for k, v in WORLD.items()},
                'class_counts': np.bincount(y, minlength=3).tolist(),
                'cv_accuracy': float(np.mean(predicted == y)),
                'cv_corner_accuracy': float(np.mean(predicted[corners] == y[corners]))}
    KickSidePredictor.fit(x, y, metadata=metadata).save(args.output)
    print('episodios %d | acierto (validación cruzada) %.1f %% | esquinas %.1f %%' % (
        len(y), 100*metadata['cv_accuracy'], 100*metadata['cv_corner_accuracy']))
    print('predictor: %s' % args.output)


if __name__ == '__main__':
    main()
