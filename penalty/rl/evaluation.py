"""Episode rollouts and multi-seed, multi-process benchmarks."""
from concurrent.futures import ProcessPoolExecutor
import time

from penalty.rl.metrics import summarize


def make_env(role, backend='synthetic', opponent=None, backend_options=None, reward_id=None):
    if role == 'striker':
        from penalty.rl.striker_env import PenaltyStrikerEnv as Env
    elif role == 'keeper':
        from penalty.rl.keeper_env import PenaltyKeeperEnv as Env
    else:
        raise ValueError('rol desconocido')
    return Env(backend=backend, opponent=opponent, reward_id=reward_id,
               backend_options=backend_options)


def rollout(env, policy, seed, options=None):
    """One episode; returns its episode_record (plus the seed)."""
    observation, _ = env.reset(seed=seed, options=options)
    if hasattr(policy, 'reset'):
        policy.reset()
    while True:
        observation, _, terminated, truncated, info = env.step(policy(observation))
        if terminated or truncated:
            return dict(info['episode_record'], seed=seed)


def _worker(job):
    from penalty.rl.policies.baselines import make_policy
    env = make_env(job['role'], job['backend'], job['opponent'], job['backend_options'])
    start = time.perf_counter()
    try:
        if job['policy'] == 'random':
            # One policy per episode seed: results do not depend on worker split.
            records = [rollout(env, make_policy('random', job['role'], seed=seed), seed)
                       for seed in job['episode_seeds']]
        else:
            policy = make_policy(job['policy'], job['role'])
            records = [rollout(env, policy, seed) for seed in job['episode_seeds']]
    finally:
        env.close()
    return records, time.perf_counter() - start


def evaluate(role, policy, backend='synthetic', opponent=None, episodes=100, seeds=(0,),
             workers=1, backend_options=None):
    """Runs `episodes` per seed; seed s yields episode seeds s*1_000_000 + i."""
    jobs = []
    for seed in seeds:
        episode_seeds = [seed*1_000_000 + i for i in range(episodes)]
        chunks = max(1, min(workers, episodes))
        for k in range(chunks):
            jobs.append({'role': role, 'backend': backend, 'opponent': opponent, 'policy': policy,
                         'backend_options': dict(backend_options or {}, **({'seed': seed} if backend == 'bhuman' else {})),
                         'episode_seeds': episode_seeds[k::chunks], 'seed': seed})
    start = time.perf_counter()
    if workers <= 1:
        results = [_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_worker, jobs))
    wall = time.perf_counter() - start
    by_seed = {}
    for job, (records, _) in zip(jobs, results):
        by_seed.setdefault(job['seed'], []).extend(records)
    everything = [r for records in by_seed.values() for r in records]
    steps = sum(r['steps'] for r in everything)
    return {'summary': summarize(everything, role),
            'per_seed': {str(s): summarize(r, role) for s, r in by_seed.items()},
            'throughput': {'workers': workers, 'wall_s': wall, 'episodes': len(everything),
                           'episodes_per_s': len(everything)/wall if wall else None,
                           'decisions_per_s': steps/wall if wall else None},
            'records': everything}
