"""PPO curriculum for the tactical striker and keeper (stable-baselines3).

Stages run in order: every synthetic stage (fast, fixed timesteps) for both
roles, then B-Human stages that share the remaining wall-clock budget. Each
role keeps one policy that is carried from stage to stage. Checkpoints are
written every few minutes and a run can be resumed after a crash or reboot.
"""
from collections import Counter, deque
import ctypes
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np

from penalty.rl.evaluation import evaluate, make_env
from penalty.rl.run_metadata import run_metadata

ROLES = ('striker', 'keeper')
HISTORY = {'striker': 1, 'keeper': 6}
CHECKPOINT_S = 600


@dataclass
class Stage:
    name: str
    role: str
    backend: str
    opponent: str
    timesteps: int = 0          # synthetic: fixed budget
    share: float = 0.           # bhuman: fraction of the remaining hours
    backend_options: dict = field(default_factory=dict)
    eval_episodes: int = 500


def overnight_plan(roles=ROLES):
    """Curricula from docs/rl/06_SELF_PLAY_AND_CURRICULUM.md (no self-play yet)."""
    stages = []
    if 'striker' in roles:
        stages += [Stage('striker_synth_empty', 'striker', 'synthetic', 'empty', 500_000),
                   Stage('striker_synth_static', 'striker', 'synthetic', 'static', 1_000_000),
                   Stage('striker_synth_random', 'striker', 'synthetic', 'random', 1_500_000),
                   Stage('striker_synth_scripted', 'striker', 'synthetic', 'scripted', 2_000_000)]
    if 'keeper' in roles:
        stages += [Stage('keeper_synth_easy', 'keeper', 'synthetic', 'synthetic', 1_500_000,
                         backend_options={'speed_range': (1.5, 2.5), 'target_range': (-.6, .6)}),
                   Stage('keeper_synth_full', 'keeper', 'synthetic', 'synthetic', 3_500_000)]
    bhuman = []
    if 'striker' in roles:
        bhuman += [Stage('striker_bhuman_static', 'striker', 'bhuman', 'static', share=.4, eval_episodes=24),
                   Stage('striker_bhuman_scripted', 'striker', 'bhuman', 'scripted', share=.6, eval_episodes=24)]
    if 'keeper' in roles:
        bhuman += [Stage('keeper_bhuman_random_kicks', 'keeper', 'bhuman', 'random_kicks', share=1.,
                         eval_episodes=24)]
    roles_with_bhuman = {s.role for s in bhuman}
    for stage in bhuman:  # shares are per role; split hours evenly across roles
        stage.share /= len(roles_with_bhuman)
    return stages + bhuman


PPO_SETTINGS = {
    'synthetic': {'n_envs': 8, 'n_steps': 512, 'batch_size': 512, 'learning_rate': 3e-4,
                  'ent_coef': .01},
    'bhuman': {'n_steps': 128, 'batch_size': 256, 'learning_rate': 1e-4, 'ent_coef': .005},
}


class ResilientEnv(gym.Wrapper):
    """Retries a failed B-Human reset with a fresh session instead of
    killing an overnight run. Step failures are already terminal
    `sim_error` episodes inside BHumanWorld."""

    def __init__(self, env, attempts=5):
        super().__init__(env)
        self.attempts = attempts
        self.failures = 0

    def reset(self, **kwargs):
        for attempt in range(self.attempts):
            try:
                return self.env.reset(**kwargs)
            except (RuntimeError, OSError) as error:
                self.failures += 1
                print('reset falló (%d/%d): %s' % (attempt+1, self.attempts, error),
                      file=sys.stderr, flush=True)
                self.env.unwrapped.world.close()
                kwargs.pop('seed', None)  # do not replay the same failing seed
                time.sleep(1.)
        raise RuntimeError('B-Human no pudo reiniciar tras %d intentos' % self.attempts)


def build_env(role, backend, opponent, backend_options=None, seed=0):
    from penalty.rl.wrappers import ObservationHistory
    from stable_baselines3.common.monitor import Monitor
    options = dict(backend_options or {})
    if backend == 'bhuman':
        options.setdefault('seed', seed)
    env = make_env(role, backend, opponent, options)
    if HISTORY[role] > 1:
        env = ObservationHistory(env, HISTORY[role])
    if backend == 'bhuman':
        env = ResilientEnv(env)
    return Monitor(env)


def _env_factory(role, backend, opponent, backend_options, seed):
    def factory():
        env = build_env(role, backend, opponent, backend_options, seed)
        env.reset(seed=seed)
        return env
    return factory


def make_vec_env(stage, workers, seed):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    n = PPO_SETTINGS['synthetic']['n_envs'] if stage.backend == 'synthetic' else workers
    factories = [_env_factory(stage.role, stage.backend, stage.opponent, stage.backend_options,
                              seed*1000 + i) for i in range(n)]
    if stage.backend == 'synthetic':
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method='spawn')


def save_policy(model, normalizer, directory, role, extra):
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory/'model.zip')
    np.savez(directory/'obs_norm.npz', mean=normalizer.obs_rms.mean, var=normalizer.obs_rms.var,
             count=normalizer.obs_rms.count, clip_obs=normalizer.clip_obs, epsilon=normalizer.epsilon)
    config = dict(role=role, history=HISTORY[role], **extra)
    (directory/'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n',
                                         encoding='utf-8')


def load_normalizer_stats(normalizer, directory):
    stats = np.load(directory/'obs_norm.npz')
    normalizer.obs_rms.mean = stats['mean'].copy()
    normalizer.obs_rms.var = stats['var'].copy()
    normalizer.obs_rms.count = float(stats['count'])


class _Progress:
    """progress.json: completed stages and per-stage elapsed seconds."""

    def __init__(self, path):
        self.path = path
        self.data = (json.loads(path.read_text(encoding='utf-8')) if path.is_file()
                     else {'completed': [], 'elapsed': {}, 'reports': {}})

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        tmp.replace(self.path)


def _callback(stage, deadline, progress, checkpoint):
    from stable_baselines3.common.callbacks import BaseCallback

    class StageCallback(BaseCallback):
        """Wall-clock deadline, outcome rates and periodic checkpoints."""

        def _on_training_start(self):
            self.outcomes = deque(maxlen=500)
            self.last_checkpoint = time.monotonic()
            self.started = time.monotonic()
            self.base_elapsed = progress.data['elapsed'].get(stage.name, 0.)

        def _on_step(self):
            for info in self.locals.get('infos', ()):
                record = info.get('episode_record')
                if record:
                    self.outcomes.append(record['outcome'])
            now = time.monotonic()
            if now - self.last_checkpoint >= CHECKPOINT_S:
                self.last_checkpoint = now
                progress.data['elapsed'][stage.name] = self.base_elapsed + now - self.started
                checkpoint()
                progress.save()
            return deadline is None or now < deadline

        def _on_rollout_end(self):
            counts = Counter(self.outcomes)
            total = max(1, len(self.outcomes))
            for outcome, count in counts.items():
                self.logger.record('outcome/%s_rate' % outcome, count/total)

    return StageCallback()


def _prevent_sleep(active):
    if sys.platform != 'win32':
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if active else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def _baseline_name(role):
    return 'scripted' if role == 'striker' else 'geometric'


def evaluate_stage(stage, policy_dir, workers, seed):
    episodes = stage.eval_episodes
    seeds = (10_000 + seed,)  # held out from training seeds
    common = dict(backend=stage.backend, opponent=stage.opponent, episodes=episodes, seeds=seeds,
                  workers=1 if stage.backend == 'synthetic' else workers,
                  backend_options=stage.backend_options)
    trained = evaluate(stage.role, str(policy_dir), **common)
    baseline = evaluate(stage.role, _baseline_name(stage.role), **common)
    return {'trained': trained['summary'], 'baseline': baseline['summary'],
            'baseline_policy': _baseline_name(stage.role)}


def train(output, hours=10., roles=ROLES, workers=8, seed=0, resume=False, smoke=False):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize

    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('%s ya existe; usa --resume o otra carpeta' % output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    stages = overnight_plan(roles)
    if smoke:  # end-to-end check in minutes: tiny budgets, same code paths
        for stage in stages:
            stage.timesteps = min(stage.timesteps, 4096)
            stage.eval_episodes = 20 if stage.backend == 'synthetic' else 4
    progress = _Progress(output/'progress.json')
    if 'metadata' not in progress.data:
        progress.data['metadata'] = run_metadata(seed=seed, hours=hours, roles=list(roles),
                                                 workers=workers, algorithm='PPO',
                                                 stages=[asdict(s) for s in stages],
                                                 ppo=PPO_SETTINGS)
        progress.data['started_at'] = time.time()
        progress.save()
    end_time = progress.data['started_at'] + hours*3600
    _prevent_sleep(True)
    try:
        for stage in stages:
            if stage.name in progress.data['completed']:
                continue
            deadline = None
            if stage.backend == 'bhuman':
                bhuman_left = [s for s in stages if s.backend == 'bhuman'
                               and s.name not in progress.data['completed']]
                share = stage.share / sum(s.share for s in bhuman_left)
                remaining = end_time - time.time()
                if remaining < (60 if smoke else 300):
                    print('sin tiempo para %s; se omite' % stage.name, flush=True)
                    continue
                deadline = time.monotonic() + remaining*share
            print('== etapa %s (%s/%s)' % (stage.name, stage.backend, stage.opponent), flush=True)
            _run_stage(stage, output, progress, deadline, workers, seed,
                       PPO, VecNormalize, configure)
            policy_dir = output/stage.role/'latest'
            report = evaluate_stage(stage, policy_dir, workers, seed)
            progress.data['reports'][stage.name] = report
            progress.data['completed'].append(stage.name)
            progress.save()
            _print_report(stage, report)
    finally:
        _prevent_sleep(False)
    (output/'report.json').write_text(json.dumps(progress.data, indent=2, allow_nan=False) + '\n',
                                      encoding='utf-8')
    return progress.data


def _run_stage(stage, output, progress, deadline, workers, seed, PPO, VecNormalize, configure):
    role_dir = output/stage.role
    latest = role_dir/'latest'
    settings = dict(PPO_SETTINGS[stage.backend])
    settings.pop('n_envs', None)
    venv = make_vec_env(stage, workers, seed + len(progress.data['completed']))
    normalizer = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.)
    model = PPO('MlpPolicy', normalizer, seed=seed, device='cpu', n_epochs=10, gamma=.99,
                gae_lambda=.95, clip_range=.2, policy_kwargs={'net_arch': [128, 128]}, verbose=0,
                **settings)
    if (latest/'model.zip').is_file():
        previous = PPO.load(latest/'model.zip', device='cpu')
        model.policy.load_state_dict(previous.policy.state_dict())
        model.num_timesteps = previous.num_timesteps
        load_normalizer_stats(normalizer, latest)
    model.set_logger(configure(str(output/'logs'/stage.name), ['stdout', 'csv']))

    def checkpoint():
        save_policy(model, normalizer, latest, stage.role,
                    {'stage': stage.name, 'timesteps': int(model.num_timesteps)})

    # Synthetic stages take minutes: an interrupted one reruns from `latest`.
    # B-Human stages are bounded by the wall-clock deadline instead.
    budget = stage.timesteps if stage.backend == 'synthetic' else 10**10
    try:
        model.learn(budget, callback=_callback(stage, deadline, progress, checkpoint),
                    reset_num_timesteps=False, progress_bar=False)
    finally:
        checkpoint()
        save_policy(model, normalizer, role_dir/stage.name, stage.role,
                    {'stage': stage.name, 'timesteps': int(model.num_timesteps)})
        progress.save()
        normalizer.close()


def _print_report(stage, report):
    key = 'goal_rate' if stage.role == 'striker' else 'save_rate'
    trained, baseline = report['trained']['rates'], report['baseline']['rates']
    print('== %s: %s entrenado %.3f vs baseline %s %.3f (caídas %.3f vs %.3f)' % (
        stage.name, key, trained[key], report['baseline_policy'], baseline[key],
        trained['any_fall_rate'], baseline['any_fall_rate']), flush=True)
