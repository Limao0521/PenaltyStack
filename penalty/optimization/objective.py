"""Versioned skill objectives over episode records. Never edit a version."""
import math

KICK_OBJECTIVE_V1 = {'id': 'kick_objective_v1', 'goal': 1., 'ball_speed': .1,
                     'target_error': 1., 'fall': 2., 'kick_time': .05}
BLOCK_OBJECTIVE_V1 = {'id': 'block_objective_v1', 'save': 1., 'contact': .3,
                      'coverage': .2, 'useless_fall': .5, 'reaction_time': .3}
# A shot that never reached the line counts as missing by this much (m).
MISSING_TARGET_ERROR = 1.


def _mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return sum(values)/len(values) if values else 0.


def kick_objective(records, weights=KICK_OBJECTIVE_V1):
    """J = +goal + ball_speed - target_error - fall - duration (means)."""
    w = weights
    errors = [r['target_error'] if r.get('target_error') is not None else MISSING_TARGET_ERROR
              for r in records]
    return (w['goal']*_mean([r['outcome'] == 'goal' for r in records])
            + w['ball_speed']*_mean([r.get('ball_speed') for r in records])
            - w['target_error']*_mean(errors)
            - w['fall']*_mean([bool(r.get('fallen')) for r in records])
            - w['kick_time']*_mean([r.get('kick_time') for r in records]))


def block_objective(records, weights=BLOCK_OBJECTIVE_V1):
    """J = +save + contact + coverage - useless_fall - reaction_time.

    coverage: fraction of committed guards on the shot's side."""
    w = weights
    committed = [r for r in records if r.get('reaction_time') is not None or r.get('early_dive')]
    return (w['save']*_mean([r['outcome'] == 'save' for r in records])
            + w['contact']*_mean([bool(r.get('contact')) for r in records])
            + w['coverage']*_mean([not r.get('wrong_side') for r in committed])
            - w['useless_fall']*_mean([bool(r.get('fallen')) and r['outcome'] != 'save'
                                       for r in records])
            - w['reaction_time']*_mean([r.get('reaction_time') for r in records]))
