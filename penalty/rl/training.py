"""PPO curriculum for the tactical striker and keeper (stable-baselines3).

Stages run in order: every synthetic stage (fast, fixed timesteps) for both
roles, then B-Human stages that share the remaining wall-clock budget. A
B-Human stage also ends early when its training outcome rate converges; the
freed time goes to the stages still pending. Each role keeps one policy that
is carried from stage to stage. Checkpoints are written every few minutes
and a run can be resumed after a crash or reboot.
"""
from collections import Counter, deque
import ctypes
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import gymnasium as gym
import numpy as np

from penalty.rl.evaluation import evaluate, make_env
from penalty.rl.metrics import summarize
from penalty.rl.run_metadata import run_metadata

ROLES = ('striker', 'keeper')
HISTORY = {'striker': 1, 'keeper': 6}
PRIMARY = {'striker': 'goal_rate', 'keeper': 'save_rate'}
CHECKPOINT_S = 600
SIM_ERROR_LIMIT = .10       # per convergence window; above it the runtime is unhealthy
STALL_S = 900               # no environment step for this long: the process is hung
EXIT_ABORTED = 3            # deliberate stop: the supervisor must not restart
EXIT_STALLED = 75


class TrainingAborted(RuntimeError):
    """Unsafe to continue (unhealthy simulator); needs a human."""


class TrainingDiverged(RuntimeError):
    """Policy weights became non-finite."""


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
    window: int = 1000          # bhuman: episodes per convergence window
    patience: int = 4           # windows in a row without improvement => converged
    min_minutes: float = 60.    # never declare convergence before this
    eval_every: int = 0         # bhuman: every k-th episode deterministic; it alone feeds convergence


class ConvergenceMonitor:
    """Non-overlapping windows of training episodes. A window improves when its
    primary rate beats the best window by more than the standard error of the
    difference of two windows; `patience` windows in a row without improvement
    after `min_seconds` of training mean the stage has converged."""

    def __init__(self, role, window, patience, min_seconds, state=None):
        self.role, self.window, self.patience, self.min_seconds = role, window, patience, min_seconds
        state = state or {}
        self.best = state.get('best')
        self.stale = state.get('stale', 0)
        self.history = list(state.get('history', []))
        self.records = []

    def state(self):
        return {'best': self.best, 'stale': self.stale, 'history': self.history}

    def margin(self, rate):
        return max(.01, math.sqrt(2*rate*(1-rate)/self.window))

    def add(self, record):
        """Returns the summary of the window this record closes, else None."""
        self.records.append(record)
        if len(self.records) < self.window:
            return None
        rates = summarize(self.records, self.role)['rates']
        self.records = []
        rate = rates[PRIMARY[self.role]]
        improved = self.best is None or rate > self.best + self.margin(self.best)
        if improved:
            self.best, self.stale = rate, 0
        else:
            self.stale += 1
        entry = {'primary': rate, 'any_fall_rate': rates['any_fall_rate'],
                 'sim_error_rate': rates['sim_error_rate'], 'improved': improved}
        self.history.append(entry)
        return entry

    def converged(self, elapsed_s):
        return self.stale >= self.patience and elapsed_s >= self.min_seconds

    def regressed(self):
        """The last window is clearly worse than the best one."""
        return (self.best is not None and bool(self.history)
                and self.history[-1]['primary'] < self.best - self.margin(self.best))


def check_health(entry):
    if entry['sim_error_rate'] > SIM_ERROR_LIMIT:
        raise TrainingAborted('sim_error %.1f%% en la última ventana (límite %.0f%%): revisar el runtime'
                              % (100*entry['sim_error_rate'], 100*SIM_ERROR_LIMIT))


def finite_policy(model):
    return all(bool(p.isfinite().all()) for p in model.policy.parameters())


class Watchdog(threading.Thread):
    """Kills a process whose environments stopped stepping (a hung worker
    blocks the main thread). The supervisor restarts it from the last
    checkpoint."""

    def __init__(self, limit_s=STALL_S, exit_fn=None):
        super().__init__(daemon=True)
        self.limit_s = limit_s
        self.last = time.monotonic()
        self.stopped = threading.Event()
        self.exit_fn = exit_fn or (lambda: os._exit(EXIT_STALLED))

    def beat(self):
        self.last = time.monotonic()

    def stalled(self, now=None):
        return (now if now is not None else time.monotonic()) - self.last > self.limit_s

    def run(self):
        while not self.stopped.wait(30):
            if self.stalled():
                print('sin progreso en %d s; se termina para reiniciar desde el último checkpoint'
                      % self.limit_s, file=sys.stderr, flush=True)
                self.exit_fn()
                return


# Robustness for every policy: perception noise (PerceptionNoiseWrapper
# defaults); in B-Human also ±5 cm ball, ±5 cm/±0.1 rad start poses.
NOISE = {'perception_noise': True}
VARIATION = dict(NOISE, ball_jitter=.05, start_jitter=(.05, .1))
# Keeper opponent: 3/4 of kicks aimed at |y| 0.3-0.65 m, 1/4 at the centre,
# 1960-2900 mm reach. The original uniform kicks rewarded a fixed centre guard.
CORNER_KICKS = dict(VARIATION, shot_target=(.3, .65), center_shots=.25, shot_power=(.5, 1.))
# B-Human: 100 held-out evaluation episodes; convergence on windows of 200
# deterministic episodes, one in every 5 each worker plays.
BHUMAN_EVAL = {'eval_episodes': 100, 'window': 200, 'eval_every': 5}


def overnight_plan(roles=ROLES):
    """Curricula from docs/rl/06_SELF_PLAY_AND_CURRICULUM.md (no self-play yet)."""
    stages = []
    if 'striker' in roles:
        stages += [Stage('striker_synth_empty', 'striker', 'synthetic', 'empty', 500_000,
                         backend_options=dict(NOISE)),
                   Stage('striker_synth_static', 'striker', 'synthetic', 'static', 1_000_000,
                         backend_options=dict(NOISE)),
                   Stage('striker_synth_random', 'striker', 'synthetic', 'random', 1_500_000,
                         backend_options=dict(NOISE)),
                   Stage('striker_synth_scripted', 'striker', 'synthetic', 'scripted', 2_000_000,
                         backend_options=dict(NOISE))]
    if 'keeper' in roles:
        stages += [Stage('keeper_synth_easy', 'keeper', 'synthetic', 'synthetic', 1_500_000,
                         backend_options=dict(NOISE, speed_range=(1.5, 2.5), target_range=(-.6, .6))),
                   Stage('keeper_synth_full', 'keeper', 'synthetic', 'synthetic', 3_500_000,
                         backend_options=dict(NOISE))]
    bhuman = []
    if 'striker' in roles:
        bhuman += [Stage('striker_bhuman_static', 'striker', 'bhuman', 'static', share=.4,
                         backend_options=dict(VARIATION), **BHUMAN_EVAL),
                   Stage('striker_bhuman_scripted', 'striker', 'bhuman', 'scripted', share=.6,
                         backend_options=dict(VARIATION), **BHUMAN_EVAL)]
    if 'keeper' in roles:
        bhuman += [Stage('keeper_bhuman_random_kicks', 'keeper', 'bhuman', 'random_kicks', share=1.,
                         backend_options=dict(CORNER_KICKS), **BHUMAN_EVAL)]
    roles_with_bhuman = {s.role for s in bhuman}
    for stage in bhuman:  # shares are per role; split hours evenly across roles
        stage.share /= len(roles_with_bhuman)
    return stages + bhuman


PPO_SETTINGS = {
    'synthetic': {'n_envs': 8, 'n_steps': 512, 'batch_size': 512, 'learning_rate': 3e-4,
                  'ent_coef': .01, 'target_kl': .03},
    # B-Human: asynchronous workers (penalty.rl.async_ppo); `rollout` transitions
    # per PPO update across all of them. No entropy bonus and std <= max_std:
    # with actions clipped to [-1, 1] the bonus inflated the noise until
    # training episodes were near random (std 0.7-20 in ppo_v1_20261009).
    'bhuman': {'rollout': 4096, 'batch_size': 256, 'learning_rate': 1e-4, 'ent_coef': 0.,
               'target_kl': .03, 'max_std': .5},
}
# Same observation/action spaces as the B-Human role; the learner's placeholder.
PLACEHOLDER_OPPONENT = {'striker': 'empty', 'keeper': 'synthetic'}


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


def build_env(role, backend, opponent, backend_options=None, seed=0, monitor=True):
    """monitor=False keeps stable-baselines3 (and torch) out of rollout workers."""
    from penalty.rl.wrappers import ObservationHistory
    options = dict(backend_options or {})
    if backend == 'bhuman':
        options.setdefault('seed', seed)
    env = make_env(role, backend, opponent, options)
    if HISTORY[role] > 1:
        env = ObservationHistory(env, HISTORY[role])
    if backend == 'bhuman':
        env = ResilientEnv(env)
    if not monitor:
        return env
    from stable_baselines3.common.monitor import Monitor
    return Monitor(env)


def _env_factory(role, backend, opponent, backend_options, seed):
    def factory():
        env = build_env(role, backend, opponent, backend_options, seed)
        env.reset(seed=seed)
        return env
    return factory


def make_vec_env(stage, seed):
    """Synthetic stages: fast analytic environments stepped in-process."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([_env_factory(stage.role, stage.backend, stage.opponent, stage.backend_options,
                                     seed*1000 + i) for i in range(PPO_SETTINGS['synthetic']['n_envs'])])


def worker_specs(stage, workers, seed):
    if stage.eval_every == 1 or stage.eval_every < 0:
        raise ValueError('eval_every debe ser 0 o >= 2: con 1 no queda ningún episodio para entrenar')
    return [{'role': stage.role, 'backend': stage.backend, 'opponent': stage.opponent,
             'backend_options': stage.backend_options, 'seed': seed*1000 + i,
             'eval_every': stage.eval_every} for i in range(workers)]


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


def _callback(stage, deadline, progress, checkpoint, monitor=None, save_best=None, heartbeat=None):
    from stable_baselines3.common.callbacks import BaseCallback

    class StageCallback(BaseCallback):
        """Wall-clock deadline, convergence, health checks, outcome rates and
        periodic checkpoints."""

        stop_reason = None

        def _on_training_start(self):
            self.outcomes = deque(maxlen=500)
            self.last_checkpoint = time.monotonic()
            self.started = time.monotonic()
            self.base_elapsed = progress.data['elapsed'].get(stage.name, 0.)

        def _elapsed(self, now):
            return self.base_elapsed + now - self.started

        def _on_step(self):
            if heartbeat:
                heartbeat()
            for info in self.locals.get('infos', ()):
                record = info.get('episode_record')
                if record:
                    self.outcomes.append(record['outcome'])
                sample = info.get('eval_record') if stage.eval_every else record
                entry = monitor.add(sample) if monitor and sample else None
                if entry:
                    self._log_window(entry)
                    check_health(entry)
                    if entry['improved'] and save_best:
                        save_best()
            now = time.monotonic()
            if now - self.last_checkpoint >= CHECKPOINT_S:
                self.last_checkpoint = now
                progress.data['elapsed'][stage.name] = self._elapsed(now)
                checkpoint()
                progress.save()
            if monitor and monitor.converged(self._elapsed(now)):
                self.stop_reason = 'converged'
                return False
            if deadline is not None and now >= deadline:
                self.stop_reason = 'deadline'
                return False
            return True

        def _log_window(self, entry):
            print('   ventana %d%s: %s %.3f (mejor %.3f, sin mejora %d/%d) caídas %.3f sim_error %.3f' % (
                len(monitor.history), ' (sin ruido)' if stage.eval_every else '',
                PRIMARY[stage.role], entry['primary'], monitor.best,
                monitor.stale, monitor.patience, entry['any_fall_rate'], entry['sim_error_rate']),
                flush=True)
            self.logger.record('convergence/window_primary', entry['primary'])
            self.logger.record('convergence/best_primary', monitor.best)
            self.logger.record('convergence/stale_windows', monitor.stale)

        def _on_rollout_end(self):
            if not finite_policy(self.model):
                raise TrainingDiverged('pesos no finitos tras la actualización PPO')
            counts = Counter(self.outcomes)
            total = max(1, len(self.outcomes))
            for outcome, count in counts.items():
                self.logger.record('outcome/%s_rate' % outcome, count/total)

        def _on_training_end(self):
            progress.data['elapsed'][stage.name] = self._elapsed(time.monotonic())

    return StageCallback()


def _prevent_sleep(active):
    if sys.platform != 'win32':
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if active else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


# The first one sets the docs/RL.md gates. 'center' is the fixed centre guard a
# keeper can fall back to; beating it shows the keeper reads the shot.
BASELINES = {'striker': ('scripted',), 'keeper': ('geometric', 'center')}


def evaluate_stage(stage, policy_dir, workers, seed):
    episodes = stage.eval_episodes
    seeds = (10_000 + seed,)  # held out from training seeds
    common = dict(backend=stage.backend, opponent=stage.opponent, episodes=episodes, seeds=seeds,
                  workers=1 if stage.backend == 'synthetic' else workers,
                  backend_options=stage.backend_options)
    trained = evaluate(stage.role, str(policy_dir), **common)['summary']
    baselines = {name: evaluate(stage.role, name, **common)['summary'] for name in BASELINES[stage.role]}
    primary, *others = BASELINES[stage.role]
    gates = acceptance_gates(stage.role, trained['rates'], baselines[primary]['rates'])
    key = PRIMARY[stage.role]
    for name in others:
        gates['primary_ge_%s' % name] = trained['rates'][key] >= baselines[name]['rates'][key]
    return {'trained': trained, 'baseline': baselines[primary], 'baseline_policy': primary,
            'other_baselines': {name: baselines[name] for name in others}, 'gates': gates}


def acceptance_gates(role, trained, baseline):
    """docs/RL.md: primary rate >= baseline and falls <= baseline + 5 %."""
    key = PRIMARY[role]
    return {'primary_ge_baseline': trained[key] >= baseline[key],
            'falls_le_baseline_plus_5pct': trained['any_fall_rate'] <= baseline['any_fall_rate'] + .05}


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
            stage.window, stage.patience, stage.min_minutes = 16, 2, 0.
    progress = _Progress(output/'progress.json')
    if 'metadata' not in progress.data:
        progress.data['metadata'] = run_metadata(seed=seed, hours=hours, roles=list(roles),
                                                 workers=workers, algorithm='PPO',
                                                 stages=[asdict(s) for s in stages],
                                                 ppo=PPO_SETTINGS)
        progress.data['started_at'] = time.time()
        progress.save()
    elif resume:
        progress.data.setdefault('resumes', []).append(
            {'at': time.time(), 'hours': hours, 'workers': workers, 'ppo': PPO_SETTINGS,
             'metadata': run_metadata(seed=seed, hours=hours, roles=list(roles), workers=workers)})
        progress.save()
    end_time = progress.data['started_at'] + hours*3600
    _prevent_sleep(True)
    watchdog = Watchdog()
    watchdog.start()
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
            watchdog.beat()
            result = _run_stage(stage, output, progress, deadline, workers, seed,
                                PPO, VecNormalize, configure, watchdog.beat)
            watchdog.beat()
            policy_dir = output/stage.role/'latest'
            report = dict(evaluate_stage(stage, policy_dir, workers, seed), training=result)
            watchdog.beat()
            progress.data['reports'][stage.name] = report
            progress.data['completed'].append(stage.name)
            progress.save()
            _print_report(stage, report)
    except TrainingAborted as error:
        progress.data['aborted'] = str(error)
        progress.save()
        raise
    finally:
        watchdog.stopped.set()
        _prevent_sleep(False)
    (output/'report.json').write_text(json.dumps(progress.data, indent=2, allow_nan=False) + '\n',
                                      encoding='utf-8')
    return progress.data


def _log_dir(directory):
    """SB3 rewrites progress.csv: a resumed stage logs to resume<k>/."""
    candidate, k = directory, 0
    while (candidate/'progress.csv').exists():
        k += 1
        candidate = directory/('resume%d' % k)
    return candidate


def _run_stage(stage, output, progress, deadline, workers, seed, PPO, VecNormalize, configure,
               heartbeat=None):
    role_dir = output/stage.role
    latest = role_dir/'latest'
    best_dir = role_dir/(stage.name + '_best')
    monitor = None
    if stage.backend == 'bhuman':
        monitor = ConvergenceMonitor(stage.role, stage.window, stage.patience, stage.min_minutes*60,
                                     progress.data.setdefault('convergence', {}).get(stage.name))
    settings = dict(PPO_SETTINGS[stage.backend])
    settings.pop('n_envs', None)
    max_std = settings.pop('max_std', None)
    run_seed = seed + len(progress.data['completed'])
    algorithm, collector = PPO, None
    if stage.backend == 'bhuman':
        from stable_baselines3.common.vec_env import DummyVecEnv
        from penalty.rl.async_ppo import AsyncCollector, AsyncPPO as algorithm
        settings['n_steps'] = settings.pop('rollout')
        # Spaces and observation statistics only: the workers own B-Human.
        venv = DummyVecEnv([_env_factory(stage.role, 'synthetic', PLACEHOLDER_OPPONENT[stage.role],
                                         {}, run_seed)])
    else:
        venv = make_vec_env(stage, run_seed)
    normalizer = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.)
    model = algorithm('MlpPolicy', normalizer, seed=seed, device='cpu', n_epochs=10, gamma=.99,
                      gae_lambda=.95, clip_range=.2, policy_kwargs={'net_arch': [128, 128]}, verbose=0,
                      **settings)
    if (latest/'model.zip').is_file():
        previous = PPO.load(latest/'model.zip', device='cpu')
        model.policy.load_state_dict(previous.policy.state_dict())
        model.num_timesteps = previous.num_timesteps
        load_normalizer_stats(normalizer, latest)
    model.set_logger(configure(str(_log_dir(output/'logs'/stage.name)), ['stdout', 'csv']))

    def extra():
        return {'stage': stage.name, 'timesteps': int(model.num_timesteps)}

    def checkpoint():
        # Never overwrite the last good checkpoint with diverged weights.
        if not finite_policy(model):
            return
        save_policy(model, normalizer, latest, stage.role, extra())
        if monitor:
            progress.data['convergence'][stage.name] = monitor.state()

    def save_best():
        if finite_policy(model):
            save_policy(model, normalizer, best_dir, stage.role,
                        dict(extra(), window_primary=monitor.best))

    # Synthetic stages take minutes: an interrupted one reruns from `latest`.
    # B-Human stages end on convergence or at the wall-clock deadline.
    budget = stage.timesteps if stage.backend == 'synthetic' else 10**10
    callback = _callback(stage, deadline, progress, checkpoint, monitor, save_best, heartbeat)
    stop = 'budget'
    try:
        if stage.backend == 'bhuman':
            model.max_std = max_std
            collector = model.collector = AsyncCollector(worker_specs(stage, workers, run_seed))
        model.learn(budget, callback=callback, reset_num_timesteps=False, progress_bar=False)
        stop = callback.stop_reason or 'budget'
    except TrainingDiverged:
        stop = 'diverged'
    except ValueError:
        # torch distributions reject NaN parameters before our check runs.
        if finite_policy(model):
            raise
        stop = 'diverged'
    finally:
        if collector is not None:
            collector.close()
        checkpoint()
        if finite_policy(model):
            save_policy(model, normalizer, role_dir/stage.name, stage.role, extra())
        progress.save()
        normalizer.close()
    restored = False
    if (best_dir/'model.zip').is_file() and (stop == 'diverged' or (monitor and monitor.regressed())):
        shutil.copytree(best_dir, latest, dirs_exist_ok=True)
        restored = True
    if stop == 'diverged':
        print('   %s divergió; se continúa desde el último checkpoint válido' % stage.name,
              file=sys.stderr, flush=True)
    return {'stop': stop, 'restored_best': restored, 'timesteps': int(model.num_timesteps),
            'windows': monitor.history if monitor else None}


def _print_report(stage, report):
    key = PRIMARY[stage.role]
    trained, baseline = report['trained']['rates'], report['baseline']['rates']
    training = report['training']
    gates = 'OK' if all(report['gates'].values()) else 'FALLA %s' % [
        name for name, ok in report['gates'].items() if not ok]
    others = ''.join(', %s %.3f' % (name, summary['rates'][key])
                     for name, summary in report.get('other_baselines', {}).items())
    print('== %s (%s%s): %s entrenado %.3f vs baseline %s %.3f%s (caídas %.3f vs %.3f) compuertas %s' % (
        stage.name, training['stop'], ', se restauró el mejor' if training['restored_best'] else '',
        key, trained[key], report['baseline_policy'], baseline[key], others,
        trained['any_fall_rate'], baseline['any_fall_rate'], gates), flush=True)
