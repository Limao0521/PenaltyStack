"""Versioned PPO fine-tuning against the frozen RGB/sensor v2 balanced keeper.

This is a separate experiment, not a change to the published curriculum,
observation or reward. SB3 performs the ordinary synchronous PPO updates;
small, torch-free subprocesses own the physical environments. Validation
selects a checkpoint before a paired blind test can start.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import hashlib
import json
import math
from multiprocessing import get_context
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np

from penalty.motion_contract import ROOT
from penalty.rl.contracts import OBSERVATION_VERSION, STRIKER_ACTION_VERSION, OBS_LOW
from penalty.rl.metrics import wilson_interval

TRAINING_VERSION = 'striker_visual_retraining_v2'
EVALUATION_VERSION = 'striker_visual_crossplay_v2'
CHECKPOINT_VERSION = 'striker_visual_checkpoint_v2'
INITIALIZATION_VERSION = 'striker_policy_init_v3'
REWARD_VERSION = 'striker_visual_reward_v3'
SPLIT_BASES = {'train': 101001, 'validation': 102001, 'test': 103001}
POLICY_FILES = ('model.zip', 'obs_norm.npz', 'config.json')
SOURCE_PATHS = ('penalty/rl/striker_retraining.py', 'penalty/rl/striker_visual_env.py',
                'penalty/rl/training.py', 'penalty/rl/rewards.py',
                'tools/retrain_striker.py', 'penalty/rl/motion_world.py',
                'penalty/rl/motion_training.py', 'penalty/skills/keeper_motion.py',
                'native/bhuman_motion/main.cpp', 'native/bhuman_motion/factories.h',
                'native/bhuman_motion/keyframe_factory.cpp', 'penalty/rl/striker_calibration.py',
                'tools/calibrate_striker_retraining.py',
                'penalty/rl/striker_retraining_v2.py', 'penalty/rl/striker_visual_env_v2.py',
                'penalty/rl/striker_preserving_policy.py', 'tools/retrain_striker_v2.py',
                'native/bhuman_motion/build/vision/Release/onnxruntime.dll')
LIMITATIONS = (
    'MuJoCo B-Human physics only, no physical NAO validation. The keeper uses '
    'RGB ball perception and simulated encoder/IMU/FSR sensors; physics pauses '
    'during RGB inference. The striker still receives rl_obs_v1 privileged '
    'scene measurements with PerceptionNoise v1, as striker_v2 did. '
    'The opponent is our v2 balanced keeper, not a complete official B-Human '
    'team. This bounded fine-tuning pilot does not establish convergence. '
    'Original observation normalization remains frozen during this pilot. '
    'Resuming restores PPO/optimizer/RNG checkpoints but starts fresh physical '
    'episodes with previously unused training seeds; no exact simulator-state replay.'
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def episode_seed(split, index):
    if split not in SPLIT_BASES or type(index) is not int or not 0 <= index < 1_000_000:
        raise ValueError('unknown split or episode index outside [0, 1000000)')
    return SPLIT_BASES[split]*1_000_000+index


def policy_hashes(directory):
    directory = Path(directory)
    return {name: _sha(directory/name) for name in POLICY_FILES}


def validate_base(directory):
    directory = Path(directory)
    config = _read(directory/'config.json')
    if config.get('role') != 'striker' or config.get('history', 1) != 1:
        raise ValueError('warm start requires a striker policy with history=1')
    with np.load(directory/'obs_norm.npz', allow_pickle=False) as stats:
        expected = (len(OBS_LOW),)
        if (stats['mean'].shape != expected or stats['var'].shape != expected or
                not np.isfinite(stats['mean']).all() or not np.isfinite(stats['var']).all() or
                not (stats['var'] > 0).all() or not math.isfinite(float(stats['count'])) or
                float(stats['count']) <= 0 or not 0 < float(stats['clip_obs']) < math.inf or
                not 0 < float(stats['epsilon']) < math.inf):
            raise ValueError('warm-start normalization does not match finite rl_obs_v1')
    return policy_hashes(directory)


def sources():
    from penalty.rl.selector_training import source_metadata
    import stable_baselines3
    import torch
    metadata = source_metadata()
    for name in SOURCE_PATHS:
        metadata['hashes'][name] = _sha(ROOT/name)
    metadata['versions'].update(torch=torch.__version__, stable_baselines3=stable_baselines3.__version__)
    return metadata


def prepare_run(output, *, base_policy=ROOT/'models/rl/striker_v2',
                motion_model=ROOT/'models/vision/keeper_motion_v1.json', workers=6,
                n_steps=128, updates=96, checkpoint_every=32, validation_episodes=64,
                test_episodes=100, learning_rate=3e-5, max_seconds=7200, resume=False,
                retention_max_drop=.10):
    """Freeze sources, base weights, opponents and held-out seed sets before PPO."""
    output, base_policy, motion_model = Path(output), Path(base_policy), Path(motion_model)
    integers = (workers, n_steps, updates, checkpoint_every, validation_episodes, test_episodes)
    if (any(type(value) is not int or value <= 0 for value in integers) or
            workers > 16 or n_steps < 32 or n_steps & (n_steps-1) or
            updates > 128 or checkpoint_every > updates or
            max(validation_episodes, test_episodes) >= 1_000_000 or
            not 0 < learning_rate <= 1e-3 or not math.isfinite(max_seconds) or max_seconds <= 0 or
            not math.isfinite(retention_max_drop) or not 0 <= retention_max_drop <= 1):
        raise ValueError('invalid bounded PPO/episode configuration')
    from penalty.rl.motion_training import load_profile
    load_profile(motion_model)
    base_hashes, motion_hash = validate_base(base_policy), _sha(motion_model)
    metadata = sources()
    configuration = {'training': TRAINING_VERSION, 'observation': OBSERVATION_VERSION,
                     'action': STRIKER_ACTION_VERSION, 'reward': REWARD_VERSION,
                     'primary_keeper': 'vision_keeper_v2:balanced:original_motion',
                     'secondary_keeper': 'vision_keeper_v2:balanced:keeper_motion_v1',
                     'base_policy_sha256': base_hashes, 'motion_model_sha256': motion_hash,
                     'initialization': INITIALIZATION_VERSION,
                     'actor_weights_preserved_initially': True, 'head_rows_rescaled': False,
                     'async_training_compatible': False,
                     'action_mean_tail_compression': {'threshold': 1., 'slope': .01,
                                                      'deterministic_postclip_identity': True},
                     'value_head_reset': True, 'optimizer_reset': True, 'initial_action_std': .25,
                     'test_opponents': ['balanced', 'motion', 'scripted'],
                     'validation_opponents': ['balanced', 'scripted'],
                     'training_opponents_by_worker': (['balanced']*(workers-1)+['scripted']
                                                       if workers > 1 else ['balanced']),
                     'training_scripted_fraction': 1/workers if workers > 1 else 0.,
                     'retention_max_drop': retention_max_drop,
                     'promotion_rule': {'validation_primary_goal_rate_strictly_above_v2': True,
                                        'validation_fsm_point_drop_at_most': retention_max_drop,
                                        'blind_primary_paired_ci95_lower_strictly_above_zero': True,
                                        'blind_fsm_point_drop_at_most': retention_max_drop,
                                        'retention_is_formal_noninferiority_test': False},
                     'selection_rule': 'best trained primary validation goals satisfying FSM retention; diagnostic fallback if none; promotion compared separately with v2',
                     'scripted_opponent': 'our scene-based FSM with B-Human motions; not official B-Human behavior',
                     'normalization': 'frozen striker_v2 observation statistics; no reward normalization',
                     'workers': workers, 'updates': updates, 'checkpoint_every': checkpoint_every,
                     'exploration_seed': SPLIT_BASES['train'],
                     'validation_episodes': validation_episodes, 'test_episodes': test_episodes,
                     'split_bases': SPLIT_BASES, 'test_may_select_or_fit': False,
                     'world_options': {'ball_jitter': .05, 'start_jitter': [.05, .1]},
                     'perception_noise': True, 'history': 1,
                     'ppo': {'n_steps': n_steps, 'batch_size': min(128, n_steps*workers),
                             'n_epochs': 4, 'learning_rate': learning_rate, 'ent_coef': 0.,
                             'target_kl': .01, 'clip_range': .05, 'gamma': 1., 'gae_lambda': .95},
                     'action_std_bounds': [.12, .40],
                     'max_training_seconds': max_seconds,
                     'source_hashes': metadata['hashes'], 'runtime_versions': metadata['versions']}
    fingerprint = _digest(configuration)
    path = output/'run_manifest.json'
    if path.exists():
        existing = _read(path)
        if not resume or existing.get('fingerprint') != fingerprint or existing.get('configuration') != configuration:
            raise ValueError('resume rejected: configuration/source/runtime/base/opponent changed')
        _check_frozen(output, existing)
        return existing
    if resume:
        raise ValueError('--resume needs an existing run_manifest.json')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('new experiment output must be empty')
    frozen = output/'frozen'
    (frozen/'striker_v2').mkdir(parents=True)
    for name in POLICY_FILES:
        shutil.copy2(base_policy/name, frozen/'striker_v2'/name)
    shutil.copy2(motion_model, frozen/'keeper_motion_v1.json')
    manifest = {'version': TRAINING_VERSION, 'fingerprint': fingerprint,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'configuration': configuration, 'source': metadata,
                'test_protocol_fixed_before_training': True, 'hardware_validated': False,
                'limitations': LIMITATIONS}
    _check_frozen(output, manifest)
    _write(path, manifest)
    return manifest


def _check_frozen(output, manifest):
    configuration = manifest['configuration']
    if (policy_hashes(Path(output)/'frozen/striker_v2') != configuration['base_policy_sha256'] or
            _sha(Path(output)/'frozen/keeper_motion_v1.json') != configuration['motion_model_sha256']):
        raise ValueError('frozen base policy/opponent bytes changed')


def check_sources(output, manifest):
    current = sources()
    configuration = manifest['configuration']
    if current['hashes'] != configuration['source_hashes'] or current['versions'] != configuration['runtime_versions']:
        raise ValueError('frozen training sources/runtime changed')
    _check_frozen(output, manifest)


def _envelope(value, fingerprint):
    return {'version': CHECKPOINT_VERSION, 'fingerprint': fingerprint,
            'payload_sha256': _digest(value), 'payload': value}


def _read_envelope(path, fingerprint):
    envelope = _read(path)
    if (envelope.get('version') != CHECKPOINT_VERSION or envelope.get('fingerprint') != fingerprint or
            envelope.get('payload_sha256') != _digest(envelope.get('payload'))):
        raise ValueError('checkpoint identity/integrity differs: '+str(path))
    return envelope['payload']


def summarize(records):
    if not records:
        raise ValueError('evaluation requires nonempty records')
    counts = Counter(record['result'] for record in records)
    total = len(records)
    return {'episodes': total, 'counts': dict(sorted(counts.items())),
            'goal_rate': counts['goal']/total, 'save_rate': counts['save']/total,
            'goal_ci95': wilson_interval(counts['goal'], total),
            'sim_errors': counts['sim_error'],
            'striker_fall_rate': sum(bool(row.get('striker_fallen')) for row in records)/total,
            'denominator': 'all scheduled episodes, including misses/falls/timeout/simulation errors'}


def paired_goals(candidate, baseline, samples=10000, seed=101001):
    a, b = {r['seed']: r for r in candidate}, {r['seed']: r for r in baseline}
    if not a or len(a) != len(candidate) or len(b) != len(baseline) or set(a) != set(b):
        raise ValueError('paired goals require identical nonempty unique physical seeds')
    keys = sorted(a)
    for key in keys:
        if _physical_reset(a[key]) != _physical_reset(b[key]):
            raise ValueError('paired policy resets differ at seed '+str(key))
    differences = np.asarray([int(a[k]['result'] == 'goal')-int(b[k]['result'] == 'goal') for k in keys])
    bootstrap = np.random.default_rng(seed).choice(differences, (samples, len(keys)), replace=True).mean(axis=1)
    return {'episodes': len(keys), 'goal_gain': float(differences.mean()),
            'goal_gain_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'candidate_only_goals': int((differences == 1).sum()),
            'baseline_only_goals': int((differences == -1).sum()),
            'ci_method': 'paired bootstrap of goal indicator, all scheduled episodes'}


def _physical_reset(record):
    reset = record.get('reset_info')
    if not isinstance(reset, dict):
        raise ValueError('paired evaluation requires physical reset metadata')
    keys = ('engine_seed', 'ball_xy', 'start_offsets')
    if any(key not in reset for key in keys):
        raise ValueError('paired reset metadata is incomplete')
    return {key: reset[key] for key in keys}


def choose_checkpoint(candidates):
    """Primary validation goals, then fewer falls, then earlier checkpoint."""
    if not candidates or any(item['summary']['sim_errors'] for item in candidates):
        raise ValueError('cannot select empty/unhealthy validation candidates')
    return min(candidates, key=lambda item: (-item['summary']['goal_rate'],
                                            item['summary']['striker_fall_rate'], item['update']))


def choose_retained_checkpoint(candidates, baseline, max_drop=.10):
    """Observed FSM retention gate, not a formal non-inferiority test.

    The best trained policy is still evaluated diagnostically if all fail.
    The frozen baseline remains the promotion candidate in that case.
    """
    if not candidates or any(item['update'] <= 0 for item in candidates):
        raise ValueError('retention selection requires trained checkpoints')
    all_items = [baseline, *candidates]
    if any(item['summary']['sim_errors'] or item['retention_summary']['sim_errors']
           for item in all_items):
        raise ValueError('unhealthy primary/retention validation')
    threshold = baseline['retention_summary']['goal_rate']-max_drop
    eligible = [item for item in candidates
                if item['retention_summary']['goal_rate']+1e-12 >= threshold]
    chosen = choose_checkpoint(eligible or candidates)
    best_eligible = choose_checkpoint(eligible) if eligible else None
    promotion = (best_eligible if best_eligible is not None and
                 best_eligible['summary']['goal_rate'] > baseline['summary']['goal_rate'] else baseline)
    return chosen, promotion, bool(eligible)


def _env_factory(configuration, index, episode_start):
    from penalty.rl.striker_visual_env_v2 import StrikerVisualKeeperEnvV2
    return StrikerVisualKeeperEnvV2(split='train', worker_index=index,
                                 worker_count=configuration['workers'], episode_start=episode_start,
                                 seed_base=SPLIT_BASES['train'],
                                 keeper_mode=configuration['training_opponents_by_worker'][index],
                                 perception_noise=True, world_options=configuration['world_options'])


class PhysicalWorkerFailure(RuntimeError):
    """The worker payload has already been persisted before raising."""
    def __init__(self, payload):
        self.payload = payload
        super().__init__('physical training worker failed:\n'+payload.get('traceback', str(payload)))


def persist_failure(output, payload, lock=None):
    """Durably append each complete failure record before any resume is allowed."""
    if output is None:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def serializable(value):
        if isinstance(value, dict):
            return {str(key): serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        if isinstance(value, np.ndarray):
            return serializable(value.tolist())
        if isinstance(value, np.generic):
            return serializable(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)
    value = serializable(dict(payload, version=TRAINING_VERSION,
                              captured_utc=datetime.now(timezone.utc).isoformat()))
    encoded = _canonical(value)+b'\n'
    if lock is not None:
        lock.acquire()
    try:
        descriptor = os.open(output/'failures.jsonl', os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if lock is not None:
            lock.release()


def _native_handles(env):
    """Keep references before a failing env closes its native session."""
    handles = []
    try:
        handles.extend(getattr(env.world.sim, 'bridges', {}).values())
    except (AttributeError, RuntimeError):
        pass
    detector = getattr(getattr(env, 'camera', None), 'detector', None)
    if detector is not None:
        handles.append(detector)
    return handles


def _native_diagnostics(handles):
    result = []
    for handle in handles:
        process = getattr(handle, 'process', None)
        result.append({'kind': type(handle).__name__, 'pid': getattr(process, 'pid', None),
                       'exit_code': process.poll() if process is not None else None,
                       'stderr_tail': list(getattr(handle, '_stderr', ())),
                       'stdout': None,
                       'stdout_note': 'native stdout is a binary/JSON IPC protocol; not duplicated'})
    return result


def _env_worker(conn, factory, failure_output=None, failure_lock=None):
    """SB3 VecEnv semantics, with terminal observation and finally cleanup."""
    env = None
    episode_return, episode_length = 0., 0
    command, payload, handles = 'initialization', None, []
    stdout, stderr = StringIO(), StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            env = factory()
        conn.send(('ok', (env.observation_space, env.action_space)))
        while True:
            command, payload = conn.recv()
            record = info = None
            stdout, stderr = StringIO(), StringIO()
            handles = _native_handles(env)
            if command == 'close':
                return
            if command == 'reset':
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    observation, info = env.reset()
                episode_return, episode_length = 0., 0
                conn.send(('ok', (observation, info)))
            elif command == 'step':
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    observation, reward, terminated, truncated, info = env.step(payload)
                record = info.get('episode_record')
                if info.get('result') == 'sim_error' or (record and record.get('result') == 'sim_error'):
                    # Abort before autoreset and before PPO receives this transition.
                    raise RuntimeError('sim_error during PPO collection: '+str(info.get('sim_error')))
                episode_return += reward
                episode_length += 1
                done = terminated or truncated
                info = dict(info)
                info['TimeLimit.truncated'] = bool(truncated and not terminated)
                reset_info = None
                if done:
                    info['terminal_observation'] = observation
                    info['episode'] = {'r': episode_return, 'l': episode_length}
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        observation, reset_info = env.reset()
                    episode_return, episode_length = 0., 0
                conn.send(('ok', (observation, reward, done, info, reset_info)))
            elif command == 'get_attr':
                conn.send(('ok', getattr(env, payload)))
            elif command == 'set_attr':
                setattr(env, payload[0], payload[1])
                conn.send(('ok', None))
            elif command == 'env_method':
                name, args, kwargs = payload
                conn.send(('ok', getattr(env, name)(*args, **kwargs)))
            elif command == 'env_is_wrapped':
                conn.send(('ok', False))
            else:
                raise ValueError('unknown vector command: '+command)
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    except BaseException as error:
        failure = {'pid': os.getpid(), 'phase': 'worker', 'command': command,
                   'worker_index': getattr(env, 'worker_index', None),
                   'seed': getattr(env, 'episode_seed', None), 'exception_type': type(error).__name__,
                   'message': str(error), 'traceback': traceback.format_exc(),
                   'episode_record': locals().get('record'),
                   'last_info': locals().get('info'),
                   'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(),
                   'native_processes': _native_diagnostics(handles),
                   'error_transition_returned_to_ppo': False}
        persist_failure(failure_output, failure, failure_lock)
        try:
            conn.send(('error', failure))
        except (EOFError, OSError):
            pass
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            conn.close()


def make_vector_env(configuration, episode_start=0, timeout_s=600, factories=None, failure_output=None):
    """Ordinary SB3 VecEnv; worker IPC and shutdown have bounded waits."""
    from stable_baselines3.common.vec_env import VecEnv

    class PhysicalVecEnv(VecEnv):
        def __init__(self):
            self.conns, self.processes = [], []
            self.waiting, self.closed = False, False
            context = get_context('spawn')
            failure_lock = self.failure_lock = context.Lock()
            try:
                for factory in (factories or [partial(_env_factory, configuration, i, episode_start)
                                             for i in range(configuration['workers'])]):
                    parent, child = context.Pipe()
                    process = context.Process(target=_env_worker,
                                              args=(child, factory, failure_output, failure_lock), daemon=True)
                    process.start()
                    child.close()
                    self.conns.append(parent)
                    self.processes.append(process)
                spaces = self._receive(self.conns)
                if any(str(pair) != str(spaces[0]) for pair in spaces):
                    raise ValueError('physical worker observation/action spaces differ')
                super().__init__(len(self.conns), *spaces[0])
            except BaseException:
                self.close()
                raise

        def _receive(self, conns):
            pending, results, deadline = list(conns), {}, time.monotonic()+timeout_s
            while pending:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('physical worker IPC exceeded %.0f s' % timeout_s)
                ready = wait(pending, timeout=min(1., remaining))
                for conn in ready:
                    try:
                        kind, value = conn.recv()
                    except EOFError:
                        raise RuntimeError('physical training worker exited unexpectedly') from None
                    if kind != 'ok':
                        if isinstance(value, dict):
                            raise PhysicalWorkerFailure(value)
                        raise RuntimeError('physical training worker failed:\n'+str(value))
                    results[conn] = value
                    pending.remove(conn)
            return [results[conn] for conn in conns]

        def reset(self):
            for conn in self.conns:
                conn.send(('reset', None))
            results = self._receive(self.conns)
            self.reset_infos = [info for _, info in results]
            self._reset_seeds()
            self._reset_options()
            return np.stack([observation for observation, _ in results])

        def seed(self, seed=None):
            # PPO seeds its own exploration. Physical reset seeds use the
            # separately frozen train agenda, never the SB3 model seed.
            self._seeds = [None]*self.num_envs
            return self._seeds

        def step_async(self, actions):
            for conn, action in zip(self.conns, actions):
                conn.send(('step', action))
            self.waiting = True

        def step_wait(self):
            rows = self._receive(self.conns)
            self.waiting = False
            for index, row in enumerate(rows):
                if row[4] is not None:
                    self.reset_infos[index] = row[4]
            return (np.stack([r[0] for r in rows]), np.asarray([r[1] for r in rows], np.float32),
                    np.asarray([r[2] for r in rows], bool), [r[3] for r in rows])

        def _command(self, command, payload, indices=None):
            conns = [self.conns[index] for index in self._get_indices(indices)]
            for conn in conns:
                conn.send((command, payload))
            return self._receive(conns)

        def get_attr(self, attr_name, indices=None):
            return self._command('get_attr', attr_name, indices)

        def set_attr(self, attr_name, value, indices=None):
            self._command('set_attr', (attr_name, value), indices)

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            return self._command('env_method', (method_name, method_args, method_kwargs), indices)

        def env_is_wrapped(self, wrapper_class, indices=None):
            return self._command('env_is_wrapped', wrapper_class, indices)

        def get_images(self):
            return [None]*self.num_envs

        def close(self):
            if self.closed:
                return
            self.closed = True
            for conn in self.conns:
                try:
                    conn.send(('close', None))
                except (OSError, ValueError):
                    pass
            deadline, open_conns = time.monotonic()+20., list(self.conns)
            while open_conns and any(p.is_alive() for p in self.processes) and time.monotonic() < deadline:
                for conn in wait(open_conns, timeout=.2):
                    try:
                        conn.recv()  # drain in-flight responses so finally can run
                    except (EOFError, OSError):
                        open_conns.remove(conn)
            import psutil
            for process in self.processes:
                process.join(timeout=.5)
                if process.is_alive():
                    try:
                        descendants = psutil.Process(process.pid).children(recursive=True)
                    except psutil.Error:
                        descendants = []
                    for child in reversed(descendants):
                        try:
                            child.terminate()
                        except psutil.Error:
                            pass
                    psutil.wait_procs(descendants, timeout=5.)
                    for child in descendants:
                        try:
                            if child.is_running():
                                child.kill()
                        except psutil.Error:
                            pass
                    process.terminate()
                    process.join(timeout=5.)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5.)
            for conn in self.conns:
                conn.close()

    return PhysicalVecEnv()


def _rng_snapshot():
    import torch
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': bytes(torch.get_rng_state().tolist()).hex()}


def _rng_restore(state):
    import torch
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, (tuple, list)) else value
    random.setstate(tuples(state['python']))
    kind, keys, position, has_gauss, cached = state['numpy']
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(torch.tensor(list(bytes.fromhex(state['torch'])), dtype=torch.uint8))


def load_warm_start(directory, vector, configuration, *, initialize=False):
    """One frozen obs transform; fresh preserving init or exact checkpoint reload."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    validate_base(directory)
    with np.load(Path(directory)/'obs_norm.npz', allow_pickle=False) as stats:
        normalizer = VecNormalize(vector, training=False, norm_obs=True, norm_reward=False,
                                  clip_obs=float(stats['clip_obs']), epsilon=float(stats['epsilon']),
                                  gamma=configuration['ppo']['gamma'])
        normalizer.obs_rms.mean = stats['mean'].copy()
        normalizer.obs_rms.var = stats['var'].copy()
        normalizer.obs_rms.count = float(stats['count'])
    try:
        if initialize:
            from penalty.rl.striker_preserving_policy import initialize_preserving_model
            model, proof = initialize_preserving_model(Path(directory)/'model.zip', normalizer,
                                                       seed=configuration.get('exploration_seed', SPLIT_BASES['train']),
                                                       **configuration['ppo'])
            model._preserving_initialization_proof = proof
        else:
            model = PPO.load(Path(directory)/'model.zip', env=normalizer, device='cpu',
                             **configuration['ppo'])
        if model.observation_space.shape != (len(OBS_LOW),) or model.action_space.shape != (4,):
            raise ValueError('warm-start model spaces differ from published striker contract')
        # Old rollout summaries are not observations from this experiment.
        model.ep_info_buffer = model.ep_success_buffer = None
        return model, normalizer
    except BaseException:
        normalizer.close()
        raise


def _save_policy(model, normalizer, directory, update, manifest, resume_state=None):
    from penalty.rl.training import finite_policy, save_policy
    if not finite_policy(model):
        raise ValueError('nonfinite PPO weights cannot be checkpointed')
    directory = Path(directory)
    if directory.exists():
        raise ValueError('immutable checkpoint already exists: '+str(directory))
    temporary = directory.with_name(directory.name+'.tmp-'+str(os.getpid())+'-'+str(time.time_ns()))
    save_policy(model, normalizer, temporary, 'striker',
                {'stage': TRAINING_VERSION, 'update': update, 'timesteps': int(model.num_timesteps),
                 'training_fingerprint': manifest['fingerprint'], 'reward': REWARD_VERSION,
                 'opponent': manifest['configuration']['primary_keeper'], 'hardware_validated': False})
    hashes = policy_hashes(temporary)
    if resume_state is not None:
        resume_state['checkpoint_sha256'] = hashes
        _write(temporary/'resume_state.json', _envelope(resume_state, manifest['fingerprint']))
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(directory)
    return hashes


def _evaluation_case(job):
    from penalty.rl.duel import make_player
    from penalty.rl.striker_visual_env_v2 import StrikerVisualKeeperEnvV2
    env = StrikerVisualKeeperEnvV2(split=job['split'], perception_noise=True,
                                seed_base=SPLIT_BASES[job['split']],
                                keeper_mode=job['keeper_mode'],
                                motion_model=job['motion_model'] if job['keeper_mode'] == 'motion' else None,
                                world_options=job['world_options'])
    records, children = {}, {}
    try:
        for name, spec in job['policies'].items():
            player = make_player(spec, 'striker')
            player.reset()
            observation, _ = env.reset(seed=job['seed'])
            children.update({id(child): child for child in env._processes()})
            finished = False
            steps = 0
            while not finished:
                observation, _, terminated, truncated, info = env.step(player(observation))
                children.update({id(child): child for child in env._processes()})
                finished = terminated or truncated
                steps += 1
                if steps > 500:
                    raise RuntimeError('bounded evaluation exceeded 500 decisions')
            records[name] = info['episode_record']
            if records[name].get('seed') != job['seed']:
                raise ValueError('evaluation physical seed differs')
    finally:
        env.close()
    children_collected = all(child.poll() is not None for child in children.values())
    if not children_collected:
        raise RuntimeError('evaluation did not collect every owned native child')
    return {'split': job['split'], 'seed': job['seed'], 'keeper_mode': job['keeper_mode'],
            'records': records, 'children_collected': children_collected,
            'owned_native_children': len(children)}


def evaluate_policies(output, manifest, split, policies, *, modes=('balanced',), workers=4,
                      tag='evaluation', progress=print):
    """Cache by exact policy weights/stats and frozen physical episode identity."""
    from penalty.rl.duel import export_trained
    output = Path(output)
    count = manifest['configuration'][split+'_episodes']
    specs = {name: export_trained(path) for name, path in policies.items()}
    hashes = {name: policy_hashes(path) for name, path in policies.items()}
    identity = {'run': manifest['fingerprint'], 'evaluation': EVALUATION_VERSION,
                'split': split, 'policies': hashes, 'modes': list(modes), 'episodes': count}
    fingerprint = _digest(identity)
    cache_dir = output/'evaluations'/tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs, rows = [], []
    for mode in modes:
        for index in range(count):
            seed = episode_seed(split, index)
            path = cache_dir/(mode+'-'+str(seed)+'.json')
            if path.exists():
                row = _read_envelope(path, fingerprint)
                if row.get('seed') != seed or row.get('split') != split or row.get('keeper_mode') != mode:
                    raise ValueError('cached evaluation episode identity differs')
                rows.append(row)
            else:
                jobs.append({'split': split, 'seed': seed, 'keeper_mode': mode,
                             'motion_model': str(output/'frozen/keeper_motion_v1.json'),
                             'world_options': manifest['configuration']['world_options'], 'policies': specs})
    started = time.monotonic()
    if jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as executor:
            futures = {executor.submit(_evaluation_case, job): job for job in jobs}
            try:
                for future in as_completed(futures):
                    row = future.result()
                    path = cache_dir/(row['keeper_mode']+'-'+str(row['seed'])+'.json')
                    _write(path, _envelope(row, fingerprint))
                    rows.append(row)
                    progress('%s: %d/%d physical cases (%.1f s)' %
                             (tag, len(rows), count*len(modes), time.monotonic()-started))
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    groups = {mode: {name: [] for name in policies} for mode in modes}
    for row in sorted(rows, key=lambda item: (item['keeper_mode'], item['seed'])):
        if set(row['records']) != set(policies):
            raise ValueError('cached evaluation policy set differs')
        for name, record in row['records'].items():
            groups[row['keeper_mode']][name].append(record)
    result = {'version': EVALUATION_VERSION, 'fingerprint': fingerprint, 'identity': identity,
              'split': split, 'summaries': {mode: {name: summarize(group) for name, group in peers.items()}
                                          for mode, peers in groups.items()},
              'records': groups, 'rows': rows}
    if any(summary['sim_errors'] for peers in result['summaries'].values() for summary in peers.values()):
        _write(cache_dir/'report.json', result)
        raise RuntimeError('sim_error in evaluation; selection/testing stopped')
    if any(policy_hashes(path) != hashes[name] for name, path in policies.items()):
        raise RuntimeError('evaluation policy changed during physical trials')
    check_sources(output, manifest)
    _write(cache_dir/'report.json', result)
    return result


def rollout_reservation(state, configuration):
    """Prewrite a conservative seed bound before a rollout can create episodes.

    At most one reset per worker per decision, plus each worker's live reset.
    The saved next episode_start already exceeds every observed live seed.
    """
    start = state['episode_start']
    count = configuration['workers']*(configuration['ppo']['n_steps']+1)
    end = start+count
    if not 0 <= start < end < 1_000_000:
        raise ValueError('rollout seed reservation exceeds training namespace')
    return {'update': state['completed_updates']+1, 'start_index': start,
            'end_index_exclusive': end, 'reserved_indices': count,
            'resume_episode_start': end,
            'reason': 'all possible worker resets in an incomplete rollout; no simulator-state replay'}


def recover_incomplete_rollout(state):
    """Drop an unoptimized rollout; preserve checkpoint/RNG and never reuse seeds."""
    recovered = deepcopy(state)
    reservation = recovered.pop('inflight_rollout', None)
    if reservation is not None:
        recovered['episode_start'] = max(recovered['episode_start'], reservation['resume_episode_start'])
        recovered.setdefault('discarded_rollout_reservations', []).append(reservation)
    return recovered


def train(output, manifest, progress=print, *, resume_failed=False):
    """Bounded ordinary PPO updates; checkpoint only completed optimizer updates."""
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from penalty.rl.training import finite_policy, _prevent_sleep
    import torch
    output = Path(output)
    configuration = manifest['configuration']
    check_sources(output, manifest)
    if (output/'selection.json').exists():
        raise ValueError('selection frozen: further training in this run is forbidden')
    state_path = output/'progress.json'
    if state_path.exists():
        state = _read_envelope(state_path, manifest['fingerprint'])
        checkpoint = output/state['checkpoint'] if state['checkpoint'] else output/'frozen/striker_v2'
        if policy_hashes(checkpoint) != state['checkpoint_sha256']:
            raise ValueError('PPO checkpoint weights/statistics/config were modified')
    else:
        checkpoint = output/'frozen/striker_v2'
        state = {'completed_updates': 0, 'checkpoint': None,
                 'checkpoint_sha256': policy_hashes(checkpoint), 'training_transitions': 0,
                 'episode_start': 0, 'elapsed_s': 0., 'episode_records': [], 'validation': [],
                 'resumes_start_fresh_training_episodes': True}
    # A policy directory and its resume envelope are renamed atomically.
    # Recover an update finished just before progress.json could be replaced.
    for directory in sorted((output/'checkpoints').glob('update_*')):
        if not directory.is_dir() or '.tmp-' in directory.name:
            continue
        resume_path = directory/'resume_state.json'
        if not resume_path.exists():
            raise ValueError('completed checkpoint is missing its resume envelope')
        recovered = _read_envelope(resume_path, manifest['fingerprint'])
        if (recovered.get('checkpoint') != directory.relative_to(output).as_posix() or
                directory.name != 'update_%04d' % recovered.get('completed_updates', -1) or
                policy_hashes(directory) != recovered.get('checkpoint_sha256')):
            raise ValueError('completed checkpoint resume identity/bytes differ')
        if recovered['completed_updates'] > state['completed_updates']:
            state, checkpoint = recovered, directory
            _write(state_path, _envelope(state, manifest['fingerprint']))
    if state.get('inflight_rollout') or state.get('failure_status'):
        if not resume_failed:
            raise RuntimeError('incomplete/failed rollout: inspect failures.jsonl, then explicitly use --resume')
        state = recover_incomplete_rollout(state)
        state.pop('failure_status', None)
        state['explicit_failure_resume_utc'] = datetime.now(timezone.utc).isoformat()
        _write(state_path, _envelope(state, manifest['fingerprint']))
    if state['completed_updates'] >= configuration['updates'] or state['elapsed_s'] >= configuration['max_training_seconds']:
        return state
    torch.set_num_threads(2)
    vector = normalizer = model = None
    started = time.monotonic()
    latest_records, max_seed_index = [], state['episode_start']
    known_rollout_steps = 0
    update_phase = 'initialization'

    class RolloutCallback(BaseCallback):
        def _on_rollout_start(self):
            clamp_std(self.model, configuration)

        def _on_step(self):
            nonlocal max_seed_index, known_rollout_steps
            known_rollout_steps += configuration['workers']
            for info in self.locals.get('infos', ()):
                seed = info.get('episode_seed')
                if seed is not None:
                    index = int(seed)-episode_seed('train', 0)
                    if not 0 <= index < 1_000_000:
                        raise ValueError('training episode crossed the frozen training seed split')
                    max_seed_index = max(max_seed_index, index)
                record = info.get('episode_record')
                if record is not None:
                    latest_records.append(record)
                    if record['result'] == 'sim_error':
                        persist_failure(output, {'phase': 'callback', 'seed': seed,
                                                 'episode_record': record,
                                                 'traceback': ''.join(traceback.format_stack()),
                                                 'error_transition_optimized': False},
                                        getattr(vector, 'failure_lock', None))
                        raise RuntimeError('sim_error during PPO collection; stop without selecting')
            if not finite_policy(self.model):
                raise ValueError('PPO weights became nonfinite')
            return True  # complete the finite rollout; wall cap is checked between updates

    _prevent_sleep(True)
    try:
        vector = make_vector_env(configuration, state['episode_start'], failure_output=output)
        initialized = output/'initialized'
        if state['completed_updates'] == 0 and initialized.exists():
            proof_report = _read(output/'initialization_report.json')
            if (proof_report.get('fingerprint') != manifest['fingerprint'] or
                    proof_report.get('policy_sha256') != policy_hashes(initialized)):
                raise ValueError('initialized checkpoint provenance/bytes differ')
            checkpoint = initialized
            state.update(checkpoint='initialized', checkpoint_sha256=policy_hashes(initialized))
        model, normalizer = load_warm_start(checkpoint, vector, configuration,
                                            initialize=state['completed_updates'] == 0 and not initialized.exists())
        model.set_logger(configure(str(output/'logs'/('resume-update-%04d' % state['completed_updates'])),
                                   ['stdout', 'csv']))
        if 'rng' in state:
            _rng_restore(state['rng'])
        elif state['completed_updates'] == 0:
            model.set_random_seed(configuration['exploration_seed'])
            if not initialized.exists():
                proof = model._preserving_initialization_proof
                clamp_std(model, configuration)
                _save_policy(model, normalizer, initialized, 0, manifest)
                _write(output/'initialization_report.json',
                       {'version': INITIALIZATION_VERSION, 'fingerprint': manifest['fingerprint'],
                        'policy_sha256': policy_hashes(initialized), 'proof': proof,
                        'base_policy_sha256': configuration['base_policy_sha256'],
                        'normalization_sha256': _sha(initialized/'obs_norm.npz'),
                        'action_std': model.policy.log_std.detach().exp().cpu().numpy().tolist(),
                        'selected_as_trained_candidate': False})
            state['initialization_sha256'] = policy_hashes(initialized)
            state.update(checkpoint='initialized', checkpoint_sha256=policy_hashes(initialized))
            state['rng'] = _rng_snapshot()
            _write(state_path, _envelope(state, manifest['fingerprint']))
        for update in range(state['completed_updates']+1, configuration['updates']+1):
            if state['elapsed_s']+time.monotonic()-started >= configuration['max_training_seconds']:
                state['stop'] = 'wall_clock_guard'
                break
            previous_timesteps = model.num_timesteps
            state['inflight_rollout'] = rollout_reservation(state, configuration)
            _write(state_path, _envelope(state, manifest['fingerprint']))
            known_rollout_steps = 0
            update_phase = 'collect_or_optimize'
            model.learn(total_timesteps=configuration['ppo']['n_steps']*configuration['workers'],
                        callback=RolloutCallback(), reset_num_timesteps=False, progress_bar=False)
            if not finite_policy(model):
                raise ValueError('PPO optimizer produced nonfinite weights')
            clamp_std(model, configuration)
            directory = output/'checkpoints'/('update_%04d' % update)
            next_state = deepcopy(state)
            next_state.pop('inflight_rollout')
            next_state.update(completed_updates=update, checkpoint=directory.relative_to(output).as_posix(),
                              rng=_rng_snapshot(), episode_start=max_seed_index+2*configuration['workers']+1)
            next_state['training_transitions'] += int(model.num_timesteps-previous_timesteps)
            next_state['episode_records'].extend(latest_records)
            next_state['elapsed_s'] += time.monotonic()-started
            update_phase = 'saving_checkpoint'
            _save_policy(model, normalizer, directory, update, manifest, resume_state=next_state)
            # Publish the new optimizer state only after its directory and
            # recovery envelope were atomically renamed together.
            state = next_state
            latest_records.clear()
            started = time.monotonic()
            update_phase = 'publishing_progress'
            _write(state_path, _envelope(state, manifest['fingerprint']))
            update_phase = 'between_updates'
            progress('PPO update %d/%d: %d new transitions, %d completed physical episodes' %
                     (update, configuration['updates'], state['training_transitions'], len(state['episode_records'])))
        state['elapsed_s'] += time.monotonic()-started
        state['stop'] = state.get('stop', 'fixed_update_budget')
        _write(state_path, _envelope(state, manifest['fingerprint']))
    except BaseException as error:
        failure = {'phase': 'training_parent', 'exception_type': type(error).__name__,
                   'update_phase': update_phase,
                   'message': str(error), 'traceback': traceback.format_exc(),
                   'completed_updates': state['completed_updates'],
                   'checkpoint': state['checkpoint'], 'checkpoint_sha256': state['checkpoint_sha256'],
                   'discarded_completed_episode_records': latest_records,
                   'known_successful_partial_rollout_transitions': known_rollout_steps,
                   'physical_partial_rollout_transitions_exact': False,
                   'inflight_rollout': state.get('inflight_rollout'),
                   'automatic_retry': False,
                   'optimizer_update_may_have_started': (update_phase in ('collect_or_optimize', 'saving_checkpoint')
                                                         and not isinstance(error, PhysicalWorkerFailure)),
                   'previous_published_checkpoint_is_preserved': True,
                   'worker_payload': error.payload if isinstance(error, PhysicalWorkerFailure) else None}
        persist_failure(output, failure, getattr(vector, 'failure_lock', None))
        state = recover_incomplete_rollout(state)
        state.update(failure_status='aborted_training_run',
                     elapsed_s=state['elapsed_s']+time.monotonic()-started)
        _write(state_path, _envelope(state, manifest['fingerprint']))
        raise
    finally:
        try:
            if normalizer is not None:
                normalizer.close()
            elif vector is not None:
                vector.close()
        finally:
            try:
                if model is not None and hasattr(model, '_logger'):
                    model.logger.close()
            finally:
                _prevent_sleep(False)
    check_sources(output, manifest)
    _write(output/'training_report.json', {'version': TRAINING_VERSION, 'fingerprint': manifest['fingerprint'],
                                          'convergence_established': False, 'state': state,
                                          'limitations': LIMITATIONS})
    return state


def clamp_std(model, configuration):
    import torch
    low, high = configuration['action_std_bounds']
    with torch.no_grad():
        model.policy.log_std.clamp_(min=math.log(low), max=math.log(high))


def select(output, manifest, progress=print):
    output = Path(output)
    check_sources(output, manifest)
    path = output/'selection.json'
    if path.exists():
        selected = _read_envelope(path, manifest['fingerprint'])
        if policy_hashes(output/selected['policy']) != selected['policy_sha256']:
            raise ValueError('validation-selected frozen policy changed')
        return selected
    state = _read_envelope(output/'progress.json', manifest['fingerprint'])
    if state['completed_updates'] != manifest['configuration']['updates'] and state.get('stop') != 'wall_clock_guard':
        raise ValueError('cannot select before the fixed training budget or guard finishes')
    if state['completed_updates'] < 1:
        raise ValueError('no completed PPO update to validate')
    configuration = manifest['configuration']
    updates = [update for update in range(configuration['checkpoint_every'], state['completed_updates']+1,
                                         configuration['checkpoint_every'])]
    # Preregistered guard fallback: evaluate the final completed optimizer
    # update even when the wall guard stops between regular checkpoints.
    if state['completed_updates'] not in updates:
        updates.append(state['completed_updates'])
    policies = {'v2': output/'frozen/striker_v2', 'initialized': output/'initialized'}
    policies.update({'update_%04d' % update: output/'checkpoints'/('update_%04d' % update) for update in updates})
    report = evaluate_policies(output, manifest, 'validation', policies,
                               modes=tuple(configuration['validation_opponents']),
                               workers=configuration['workers'], tag='validation_selection', progress=progress)
    peers = report['summaries']['balanced']
    retention_peers = report['summaries']['scripted']
    candidates = [{'update': 0, 'policy': 'frozen/striker_v2',
                   'policy_sha256': policy_hashes(output/'frozen/striker_v2'), 'summary': peers['v2'],
                   'retention_summary': retention_peers['v2']}]
    candidates += [{'update': update, 'policy': 'checkpoints/update_%04d' % update,
                    'policy_sha256': policy_hashes(output/'checkpoints'/('update_%04d' % update)),
                    'summary': peers['update_%04d' % update],
                    'retention_summary': retention_peers['update_%04d' % update],
                    'vs_v2': paired_goals(report['records']['balanced']['update_%04d' % update],
                                          report['records']['balanced']['v2']),
                    'retention_vs_v2': paired_goals(report['records']['scripted']['update_%04d' % update],
                                                   report['records']['scripted']['v2']),
                    'vs_initialized': paired_goals(report['records']['balanced']['update_%04d' % update],
                                                   report['records']['balanced']['initialized'])} for update in updates]
    if any(policy_hashes(output/item['policy']) != item['policy_sha256'] for item in candidates):
        raise ValueError('validation candidate bytes changed')
    chosen, promotion, retained = choose_retained_checkpoint(candidates[1:], candidates[0],
                                                            configuration['retention_max_drop'])
    selected = {'version': TRAINING_VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'fingerprint': manifest['fingerprint'], 'policy': chosen['policy'],
                'policy_sha256': chosen['policy_sha256'], 'update': chosen['update'],
                'validation_summary': chosen['summary'], 'candidates': candidates,
                'retention_summary': chosen['retention_summary'],
                'retention_gate_satisfied': retained,
                'diagnostic_only': not retained,
                'retention_max_drop': configuration['retention_max_drop'],
                'promotion_candidate': promotion,
                'validation_prefers_trained_over_v2': promotion['update'] > 0,
                'initialization_diagnostic': {'summary': peers['initialized'],
                                             'retention_summary': retention_peers['initialized'],
                                             'vs_v2': paired_goals(report['records']['balanced']['initialized'],
                                                                  report['records']['balanced']['v2']),
                                             'eligible_for_selection': False},
                'validation_report_sha256': _sha(output/'evaluations/validation_selection/report.json'),
                'selection_rule': configuration['selection_rule'],
                'retention_gate_is_formal_noninferiority_test': False,
                'test_used_for_selection': False, 'hardware_validated': False}
    _write(path, _envelope(selected, manifest['fingerprint']))
    return selected


def blind_test(output, manifest, workers=None, progress=print):
    output = Path(output)
    check_sources(output, manifest)
    selected = _read_envelope(output/'selection.json', manifest['fingerprint'])
    if selected.get('test_used_for_selection') is not False:
        raise ValueError('blind test requires a validation-only selected checkpoint')
    selection_hash = _sha(output/'selection.json')
    directory = output/selected['policy']
    if policy_hashes(directory) != selected['policy_sha256']:
        raise ValueError('selected policy bytes differ')
    report = evaluate_policies(output, manifest, 'test',
                               {'v2': output/'frozen/striker_v2', 'selected': directory},
                               modes=tuple(manifest['configuration']['test_opponents']),
                               workers=workers or manifest['configuration']['workers'], tag='blind_test', progress=progress)
    report.update(run_fingerprint=manifest['fingerprint'], selection_sha256=selection_hash,
                  policy_sha256=selected['policy_sha256'], selected_update=selected['update'],
                  comparisons={mode: paired_goals(peers['selected'], peers['v2'])
                               for mode, peers in report['records'].items()},
                  model_frozen_before_test=True, test_used_for_selection=False,
                  hardware_validated=False, limitations=LIMITATIONS)
    check_sources(output, manifest)
    if _sha(output/'selection.json') != selection_hash or policy_hashes(directory) != selected['policy_sha256']:
        raise RuntimeError('frozen selection/model changed during blind test')
    _write(output/'test_report.json', report)
    return report
