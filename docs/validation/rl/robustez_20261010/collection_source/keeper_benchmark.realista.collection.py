"""Versioned, paired keeper benchmark across striker styles in B-Human.

Selection seeds and final-test seeds must be disjoint. Styles change tactical
requests only; every walk and kick is still executed by the original motor.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import math
import time

import numpy as np

from penalty.rl.contracts import FIELD, StrikerAction
from penalty.rl.duel import DuelWorld, duel_episode, make_player, summarize_duels
from penalty.rl.robust_keeper import RobustKeeperPolicy, RobustKeeperSettings
from penalty.skills.kick import NOMINAL_POWER

BENCHMARK_VERSION = 'keeper_robustness_v1'
STYLES = ('center', 'mixed', 'corners', 'soft', 'late', 'feint', 'v1', 'v2')


class VariedStrikerPolicy:
    """One reproducible style per episode, no access to simulator state."""
    def __init__(self, style, seed):
        if style not in STYLES[:6]:
            raise ValueError('estilo desconocido: ' + style)
        rng = np.random.default_rng([seed, 917])
        self.style = style
        if style == 'center':
            self.target, self.foot, self.power = 0., 0, NOMINAL_POWER
        else:
            self.target = (float(rng.uniform(-.65, .65)) if style in ('mixed', 'late') else
                           float(rng.choice((-1., 1.))*rng.uniform(.35, .65)))
            self.foot = int(rng.integers(0, 2))
            self.power = float(rng.uniform(.25, .45) if style == 'soft' else rng.uniform(.5, 1.))
        self.fire_after = float(rng.uniform(6., 10.)) if style == 'late' else 0.
        self.change_at = float(rng.uniform(1.5, 3.)) if style == 'feint' else 0.

    def __call__(self, observation):
        now = float(observation[FIELD['time_s']])
        target = -self.target if now < self.change_at else self.target
        fire = now >= max(self.fire_after, self.change_at)
        return StrikerAction(target, self.foot, self.power, float(fire)).to_normalized()


def keeper_spec(name, settings=None):
    if name == 'robust':
        return {'kind': 'robust', 'settings': asdict(settings or RobustKeeperSettings()), 'gate': False}
    from penalty.rl.duel import player_spec
    return player_spec(name)[1]


def _keeper(spec):
    return RobustKeeperPolicy(spec['settings']) if spec['kind'] == 'robust' else make_player(spec, 'keeper')


def _job(job):
    world = DuelWorld(keeper_gate=job['keeper'].get('gate', False),
                      ball_jitter=.05, start_jitter=(.05, .1))
    keeper = _keeper(job['keeper'])
    records = []
    try:
        for seed in job['seeds']:
            striker = (make_player(job['trained'][job['style']], 'striker') if job['style'] in ('v1', 'v2')
                       else VariedStrikerPolicy(job['style'], seed))
            from penalty.rl.keeper_noise import KeeperSensorStressV2
            factory = KeeperSensorStressV2 if job['condition'] == 'stress' else None
            record = duel_episode(world, striker, keeper, seed, noise=True, keeper_noise_factory=factory)
            if isinstance(keeper, RobustKeeperPolicy):
                record['keeper_decision'] = dict(keeper.decision)
            records.append(record)
    finally:
        world.close()
    return records


def aggregate(groups):
    """Equal weight per style; misses/falls/errors stay in the denominator."""
    summaries = {name: summarize_duels(records) for name, records in groups.items()}
    rates = [s['rates']['save_rate'] for s in summaries.values()]
    return {'per_style': summaries, 'macro_save_rate': float(np.mean(rates)),
            'worst_style_save_rate': min(rates),
            'worst_style': min(summaries, key=lambda s: summaries[s]['rates']['save_rate']),
            'sim_errors': sum(s['counts']['sim_error'] for s in summaries.values()),
            'early_commits': sum(bool(r.get('early_dive')) for rs in groups.values() for r in rs)}


def paired_saves(first, second):
    a, b = {r['seed']: r for r in first}, {r['seed']: r for r in second}
    if set(a) != set(b) or len(a) != len(first) or len(b) != len(second):
        raise ValueError('comparación pareada exige semillas únicas e idénticas')
    wins = sum(a[s]['result'] == 'save' and b[s]['result'] != 'save' for s in a)
    losses = sum(a[s]['result'] != 'save' and b[s]['result'] == 'save' for s in a)
    count = wins + losses
    # Exact binomial McNemar; recurrence avoids huge integer-to-float powers.
    tail = sum(math.exp(math.lgamma(count+1) - math.lgamma(k+1) - math.lgamma(count-k+1)
                        - count*math.log(2.)) for k in range(min(wins, losses)+1)) if count else .5
    return {'only_first_saves': wins, 'only_second_saves': losses, 'p_exact': min(1., 2*tail)}


def benchmark(keepers, trained, episodes=20, seed=61000, workers=8, styles=STYLES, condition='realista'):
    if episodes < 1 or workers < 1 or not styles or any(s not in STYLES for s in styles):
        raise ValueError('episodios/workers positivos y estilos conocidos requeridos')
    if condition not in ('realista', 'stress'):
        raise ValueError('condición desconocida: ' + condition)
    seeds = [seed*1_000_000 + i for i in range(episodes)]
    jobs = [{'style': style, 'name': name, 'keeper': spec, 'trained': trained,
             'condition': condition, 'seeds': seeds[k:k+5]}
            for name, spec in keepers.items() for style in styles for k in range(0, episodes, 5)]
    start = time.perf_counter()
    groups = {name: {style: [] for style in styles} for name in keepers}
    if workers == 1:
        results = map(_job, jobs)
        for job, records in zip(jobs, results):
            groups[job['name']][job['style']].extend(records)
    else:
        with ProcessPoolExecutor(workers) as pool:
            for job, records in zip(jobs, pool.map(_job, jobs)):
                groups[job['name']][job['style']].extend(records)
                count = len(groups[job['name']][job['style']])
                if count == episodes:
                    summary = summarize_duels(groups[job['name']][job['style']])
                    print('%s / %s: %d episodios, atajadas %.3f' % (
                        job['name'], job['style'], count, summary['rates']['save_rate']), flush=True)
    return {'benchmark': BENCHMARK_VERSION, 'condition': condition, 'seed': seed,
            'episode_seeds': [seeds[0], seeds[-1]],
            'episodes_per_style': episodes, 'styles': list(styles), 'keepers': keepers,
            'perception': ('PerceptionNoise v1; privileged yaw and scene-derived velocity' if condition == 'realista'
                          else 'keeper_sensor_stress_v2; synthetic yaw/velocity errors and 100 ms delay; striker noise v1'),
            'wall_s': time.perf_counter()-start,
            'summary': {name: aggregate(gs) for name, gs in groups.items()}, 'records': groups}
