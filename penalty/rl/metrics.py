"""Episode records and aggregate rates for striker and keeper evaluation."""
import math

import numpy as np

from penalty.rl.rewards import KEEPER_OUTCOMES as KEEPER_PRIMARY
from penalty.rl.rewards import STRIKER_OUTCOMES as STRIKER_PRIMARY

STRIKER_SECONDARY = ('ball_speed', 'target_error', 'kick_time')
KEEPER_SECONDARY = ('reaction_time',)
KEEPER_FLAGS = ('wrong_side', 'early_dive', 'contact', 'fallen')
SIDES = ('left', 'center', 'right')
SIDE_Y = .2  # m on the goal line; the keeper's side bins (synthetic.SIDE_THRESHOLD)


def shot_side(y):
    """Striker's view: positive y is the striker's left, i.e. the keeper's right."""
    return 'left' if y > SIDE_Y else 'right' if y < -SIDE_Y else 'center'


def side_summary(records):
    """Where the striker aimed (target_y, every kick) and where the ball crossed
    the goal line (goal_line_y, when it did), in the striker's view."""
    def rates(values):
        return {side: sum(shot_side(y) == side for y in values)/len(values) if values else 0.
                for side in SIDES}

    def scored(record):
        return (record.get('outcome') or record.get('result')) == 'goal'

    aimed = [r for r in records if r.get('target_y') is not None]
    crossed = [r['goal_line_y'] for r in records if r.get('goal_line_y') is not None]
    by_aim = {side: [r for r in aimed if shot_side(r['target_y']) == side] for side in SIDES}
    return {'kicks': len(aimed), 'crossed_goal_line': len(crossed),
            'aim': rates([r['target_y'] for r in aimed]), 'goal_line': rates(crossed),
            'goal_rate_by_aim': {side: sum(map(scored, group))/len(group) if group else None
                                 for side, group in by_aim.items()},
            'mean_abs_target_y': float(np.mean([abs(r['target_y']) for r in aimed])) if aimed else None}


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return [0., 0.]
    p = successes / total
    denominator = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., centre-half), min(1., centre+half)]


def _mean_std(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None
    return {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'n': len(values)}


def summarize(records, role):
    """records: dicts with 'outcome' plus role-specific secondary fields."""
    if role not in ('striker', 'keeper'):
        raise ValueError('rol desconocido')
    total = len(records)
    outcomes = STRIKER_PRIMARY if role == 'striker' else KEEPER_PRIMARY
    summary = {'episodes': total, 'rates': {}, 'counts': {}, 'ci95': {}}
    for outcome in outcomes:
        count = sum(r['outcome'] == outcome for r in records)
        summary['counts'][outcome] = count
        summary['rates'][outcome + '_rate'] = count/total if total else 0.
        summary['ci95'][outcome + '_rate'] = wilson_interval(count, total)
    # Falls are tracked independently of the terminal outcome.
    falls = sum(bool(r.get('fallen')) for r in records)
    summary['rates']['any_fall_rate'] = falls/total if total else 0.
    if role == 'keeper':
        for flag in KEEPER_FLAGS:
            count = sum(bool(r.get(flag)) for r in records)
            summary['rates'][flag + '_rate'] = count/total if total else 0.
    secondary = STRIKER_SECONDARY if role == 'striker' else KEEPER_SECONDARY
    summary['secondary'] = {name: _mean_std([r.get(name) for r in records]) for name in secondary}
    if role == 'striker':
        summary['sides'] = side_summary(records)
    summary['mean_return'] = _mean_std([r.get('return') for r in records])
    summary['mean_steps'] = _mean_std([r.get('steps') for r in records])
    return summary
