"""Train a six-defense selector using paired RGB/sensor B-Human trials.

The published temporal keeper v2 supplies the trajectory and trigger. Each
training context is collected six times with the same physical reset, once
per original defense. Only camera-derived context enters the learned model;
scene-derived terminal outcomes are labels and evaluation data. This is a
contextual selector, not PPO or learned joint trajectories.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import os
import subprocess
import sys
import time

import numpy as np

from penalty.keeper_defenses import ACTION_VERSION, DEFENSE_CATALOG_VERSION, DEFENSE_NAMES
from penalty.keeper_library import VisionLibraryKeeper
from penalty.keeper_selector import (CONTEXT_FIELDS, CONTEXT_VERSION, MODEL_VERSION, OBJECTIVE_VERSION, POLICY_VERSION,
                                     ForcedDefenseKeeper, LearnedDefenseKeeper,
                                     context_from_state, fit_model, load_model,
                                     save_model, seal_model, validate_model)
from penalty.motion_contract import ROOT, SOURCE_COMMIT, executable_path
from penalty.rl.contracts import StrikerAction, clip_observation
from penalty.rl.defense_benchmark import SensorDefenseDuelWorld
from penalty.rl.duel import make_player, player_spec
from penalty.rl.keeper_benchmark import STYLES, VariedStrikerPolicy, aggregate, compare_styles
from penalty.rl.wrappers import PerceptionNoise
from penalty.sim_orientation import PIPELINE_VERSION, SensorBHumanKeeperCamera


TRAINING_VERSION = 'keeper_selector_training_v1'
DATASET_VERSION = 'keeper_selector_counterfactuals_v1'
EVALUATION_VERSION = 'keeper_selector_evaluation_v1'
SPLIT_BASES = {'train': 74001, 'validation': 75001, 'test': 76001}
PAIR_TOLERANCE = 1e-9
EVALUATION_MODES = ('learned', 'original', 'balanced')
MAX_EPISODE_STEPS = 500

# Hash the new contracts and every source/model influencing observations,
# labels, physics, striker inference or native requests. Tests/docs are not
# part of cache identity, and worker count may change when resuming.
SOURCE_PATHS = (
    'penalty/keeper_selector.py', 'penalty/rl/selector_training.py',
    'tools/train_keeper_selector.py', 'penalty/keeper_defenses.py',
    'penalty/keeper_library.py', 'penalty/keeper_vision.py',
    'penalty/rl/defense_benchmark.py', 'penalty/rl/keeper_benchmark.py',
    'penalty/rl/duel.py', 'penalty/rl/bhuman_world.py',
    'penalty/rl/observation.py', 'penalty/rl/contracts.py',
    'penalty/rl/actors.py', 'penalty/rl/wrappers.py',
    'penalty/rl/async_ppo.py', 'penalty/rl/policies/trained.py',
    'penalty/rl/controllers.py', 'penalty/rl/geometry.py',
    'penalty/physical_sim.py', 'penalty/sim_session.py',
    'penalty/control.py', 'penalty/bhuman_bridge.py', 'penalty/motion_contract.py',
    'penalty/sim_sensors.py', 'penalty/sim_orientation.py',
    'penalty/sim_opponent_vision.py', 'penalty/sim_vision.py',
    'penalty/perception.py', 'penalty/vision.py', 'penalty/robot_detector.py',
    'penalty/opponent_perception.py', 'penalty/opponent_orientation.py',
    'penalty/nao_model.py', 'penalty/skills/kick.py',
    'sim/scene.xml', 'baseline/motion_manifest.json',
    'models/bhuman/NaoV6H25.rsi3',
    'models/vision/nao_yaw_v1/manifest.json',
    'models/vision/nao_yaw_v1/templates.npz',
    'heritage/bhuman_robot_perception/manifest.json',
)
LIMITATIONS = (
    'MuJoCo only; ideal own encoder/IMU/FSR readings and estimated contact odometry; '
    'physics pauses during RGB inference; no physical NAO validation. '
    'The selector inherits the v2 commitment time, even when it replaces a '
    'central guard with a jump or a lateral jump with a wide/central guard. '
    'The striker receives privileged scene observation with PerceptionNoise v1; '
    'the keeper context contains ball camera measurements and the v2 prediction only.'
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON in checkpoint: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def episode_seed(split, style, index):
    """Unique per split/style/case; paired policies share this actual seed."""
    if split not in SPLIT_BASES or style not in STYLES or type(index) is not int or not 0 <= index < 10000:
        raise ValueError('unknown split/style or invalid episode index')
    return SPLIT_BASES[split]*1_000_000+STYLES.index(style)*10_000+index


def source_metadata():
    from penalty.robot_detector import executable_path as vision_executable, model_path
    import gymnasium
    import mujoco
    from PIL import __version__ as pillow_version
    original = _read_json(ROOT/'baseline/motion_manifest.json')
    if original.get('commit') != SOURCE_COMMIT or not original.get('files'):
        raise RuntimeError('baseline motion manifest revision differs')
    for name, expected in original['files'].items():
        path = ROOT/'baseline/bhuman'/name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError('baseline B-Human file changed: '+name)
    paths = list(SOURCE_PATHS)
    for path in (executable_path(), Path(vision_executable()), Path(model_path())):
        paths.append(Path(path).resolve().relative_to(ROOT).as_posix())
    hashes = {}
    for name in paths:
        path = ROOT/name
        if not path.is_file():
            raise RuntimeError('required training source/runtime is missing: '+str(path))
        hashes[name] = _sha256(path)
    # Freeze physical MJCF/meshes and both existing striker checkpoints too.
    for directory in (ROOT/'models/bhuman', ROOT/'models/rl/striker_v1', ROOT/'models/rl/striker_v2'):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix != '.pyc':
                hashes[path.relative_to(ROOT).as_posix()] = _sha256(path)
    try:
        commit = subprocess.run(('git', 'rev-parse', 'HEAD'), cwd=ROOT, capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(('git', 'status', '--porcelain', '--untracked-files=no'), cwd=ROOT,
                                  capture_output=True, text=True, timeout=10, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    versions = {'python': sys.version.split()[0], 'numpy': np.__version__,
                'mujoco': mujoco.__version__, 'gymnasium': gymnasium.__version__,
                'pillow': pillow_version}
    return {'hashes': hashes, 'versions': versions, 'commit': commit, 'dirty': dirty,
            'platform': platform.platform(), 'bhuman_source_commit': SOURCE_COMMIT,
            'baseline_integrity': {'files': len(original['files']), 'verified': True,
                                   'manifest_sha256': hashes['baseline/motion_manifest.json']}}


def prepare_run(output, *, train_episodes=12, validation_episodes=6, test_episodes=20,
                styles=STYLES, fit_epochs=1500, resume=False):
    """Create/validate immutable run identity; execution parallelism is separate."""
    output, styles = Path(output), tuple(styles)
    counts = {'train': train_episodes, 'validation': validation_episodes, 'test': test_episodes}
    if (any(type(n) is not int or not 1 <= n <= 10000 for n in counts.values()) or
            type(fit_epochs) is not int or fit_epochs < 1 or not styles or
            len(set(styles)) != len(styles) or any(s not in STYLES for s in styles)):
        raise ValueError('positive episode counts/epochs and distinct known styles required')
    sources = source_metadata()
    configuration = {'training': TRAINING_VERSION, 'dataset': DATASET_VERSION,
                     'objective': OBJECTIVE_VERSION, 'context': CONTEXT_VERSION,
                     'action': ACTION_VERSION, 'model': MODEL_VERSION,
                     'camera_pipeline': PIPELINE_VERSION, 'styles': list(styles),
                     'episodes_per_style': counts, 'split_bases': SPLIT_BASES,
                     'seed_scheme': 'base*1000000 + canonical_style_index*10000 + episode_index',
                     'defenses': list(DEFENSE_NAMES), 'world': {'ball_jitter': .05, 'start_jitter': [.05, .1]},
                     'fit_epochs': fit_epochs, 'pair_tolerance': PAIR_TOLERANCE,
                     'source_hashes': sources['hashes'], 'runtime_versions': sources['versions']}
    fingerprint = _digest(configuration)
    manifest_path = output/'run_manifest.json'
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if not resume:
            raise ValueError('output already contains a run; use --resume or another directory')
        if existing.get('fingerprint') != fingerprint or existing.get('configuration') != configuration:
            raise ValueError('resume rejected: settings, source/model hashes or runtime versions changed')
        return existing
    if resume:
        raise ValueError('--resume requires an existing run_manifest.json')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('new run output must be empty')
    manifest = {'version': TRAINING_VERSION, 'fingerprint': fingerprint,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'configuration': configuration, 'source': sources, 'limitations': LIMITATIONS,
                'hardware_validated': False, 'learned_joint_trajectories': False}
    _atomic_json(manifest_path, manifest)
    return manifest


def _validate_capability(world):
    hello = world.sim.bridges['keeper'].hello
    if (DEFENSE_CATALOG_VERSION not in hello.get('capabilities', ()) or
            set(hello.get('keeper_defenses', ())) != set(DEFENSE_NAMES)):
        raise RuntimeError('rebuild motion runtime: six-defense capability is required')


def run_camera_episode(world, camera, striker, keeper, seed, on_frame=None):
    """One real-physics episode; no style/seed/truth fields enter the selector.

    ``on_frame(state, action, decision)`` runs before the motor action. The
    first-trigger context and timing are copied before duel_step can execute
    the defense or fast-forward to the outcome.
    """
    rng = np.random.default_rng(seed)
    striker_observation, _, reset_info = world.reset(rng, {})
    _validate_capability(world)
    camera.reset()
    keeper.reset()
    if hasattr(striker, 'reset'):
        striker.reset()
    noise, noise_rng = PerceptionNoise(), np.random.default_rng([seed, 0])
    result, steps, visible, velocities = None, 0, 0, 0
    context = timing = runtime = None
    selected_defense = None
    while result is None:
        if steps >= MAX_EPISODE_STEPS:
            raise RuntimeError('camera episode exceeded its bounded decision limit')
        state = camera.observe()
        action = keeper(state)
        decision = dict(keeper.decision)
        visible += bool(state['ball']['visible'])
        velocities += bool(state['ball']['velocity_valid'])
        if runtime is None:
            runtime = {'motion': world.sim.bridges['keeper'].hello,
                       'robot_detector': getattr(getattr(camera, 'detector', None), 'hello', None),
                       'sensor': state.get('sensor_provenance'),
                       'inference_clock': state.get('inference_clock')}
        if action.fires and context is None:
            context = context_from_state(state, decision)
            timing = {'time_s': float(state['time_s']), 'capture_s': float(state['capture_s']),
                      'core_side': int(decision['core_side']),
                      'crossing_y': float(decision['crossing_y']),
                      'time_to_line': float(decision['time_to_line'])}
            selected_defense = action.defense
        if on_frame is not None:
            on_frame(state, action, decision)
        striker_action = StrikerAction.from_normalized(
            striker(noise(clip_observation(striker_observation.to_array()), noise_rng)))
        striker_observation, _, result, _ = world.duel_step(striker_action, action)
        steps += 1
    return dict(world.record(), seed=seed, steps=steps, visible_frames=visible,
                velocity_frames=velocities, keeper_decision=dict(keeper.decision),
                selector_context=context, first_trigger=timing,
                requested_defense=selected_defense, no_commit=context is None,
                context_digest=None if context is None else _digest(context),
                reset_info=reset_info, runtime_provenance=runtime)


def _error_record(seed, error):
    return {'result': 'sim_error', 'seed': seed, 'sim_error': str(error), 'steps': 0,
            'commit_defense': None, 'selector_context': None, 'first_trigger': None,
            'context_digest': None, 'requested_defense': None, 'no_commit': True}


def _striker(style, seed, trained):
    return (make_player(trained[style], 'striker') if style in trained
            else VariedStrikerPolicy(style, seed))


def _camera(world, camera):
    world._ensure_session()
    if camera is not None and camera.model is not world.sim.model:
        camera.close()
        camera = None
    return camera if camera is not None else SensorBHumanKeeperCamera(world)


def _close_resources(world, camera):
    try:
        if camera is not None:
            camera.close()
    finally:
        world.close()


def pairing_check(records):
    """Check successful repetitions have identical pre-action context/timing.

    Every repetition must succeed and reach the same first trigger. Any
    failure or mismatch rejects the entire context from model fitting.
    """
    if set(records) != set(DEFENSE_NAMES):
        raise ValueError('pairing requires one record for each of six defenses')
    rows = [records[name] for name in DEFENSE_NAMES]
    if any(row['result'] == 'sim_error' for row in rows):
        return {'paired': False, 'reason': 'sim_error', 'max_abs_delta': 0., 'context': None}
    committed = [row for row in rows if row.get('selector_context') is not None]
    if not committed:
        return {'paired': True, 'reason': 'no_commit', 'max_abs_delta': 0., 'context': None}
    reference = committed[0]
    max_delta = 0.
    paired = len(committed) == len(rows)
    reference_values = np.asarray(reference['selector_context']['values'], dtype=float)
    reference_timing = reference['first_trigger']
    if reference['selector_context'].get('version') != CONTEXT_VERSION:
        paired = False
    for row in committed:
        context, timing = row['selector_context'], row['first_trigger']
        values = np.asarray(context['values'], dtype=float)
        if (context.get('version') != CONTEXT_VERSION or values.shape != (len(CONTEXT_FIELDS),)
                or reference_values.shape != values.shape or not np.isfinite(values).all()
                or not np.isfinite(reference_values).all()):
            paired = False
            continue
        numeric_timing = [timing[key] for key in ('time_s', 'capture_s', 'crossing_y', 'time_to_line')]
        reference_numeric = [reference_timing[key] for key in ('time_s', 'capture_s', 'crossing_y', 'time_to_line')]
        if not np.isfinite(numeric_timing+reference_numeric).all():
            paired = False
            continue
        delta = float(np.max(np.abs(values-reference_values)))
        for key in ('time_s', 'capture_s', 'crossing_y', 'time_to_line'):
            delta = max(delta, abs(timing[key]-reference_timing[key]))
        # DuelWorld records the actual commitment before stepping physics.
        # It must agree with the copied camera time as well as the other runs.
        commit_time = row.get('commit_time')
        if commit_time is None or not np.isfinite(commit_time):
            paired = False
        else:
            delta = max(delta, abs(commit_time-reference_timing['time_s']))
        max_delta = max(max_delta, delta)
        paired &= timing['core_side'] == reference_timing['core_side'] and delta <= PAIR_TOLERANCE
    for name, row in records.items():
        if row.get('selector_context') is not None:
            paired &= row.get('commit_defense') == name and row.get('requested_defense') == name
    return {'paired': bool(paired), 'reason': 'paired' if paired else 'precommit_context_mismatch',
            'max_abs_delta': max_delta, 'context': reference['selector_context'],
            'first_trigger': reference_timing,
            'exact_context_digests': {name: row.get('context_digest') for name, row in records.items()}}


def _collect_job(job):
    world = SensorDefenseDuelWorld(ball_jitter=.05, start_jitter=(.05, .1))
    camera, records = None, {}
    offset = job['seed'] % len(DEFENSE_NAMES)
    order = DEFENSE_NAMES[offset:]+DEFENSE_NAMES[:offset]
    start = time.perf_counter()
    try:
        for defense in order:
            try:
                camera = _camera(world, camera)
                records[defense] = run_camera_episode(world, camera,
                    _striker(job['style'], job['seed'], job['trained']),
                    ForcedDefenseKeeper(defense), job['seed'])
            except (RuntimeError, ValueError, OSError) as error:
                records[defense] = _error_record(job['seed'], error)
                try:
                    _close_resources(world, camera)
                finally:
                    camera = None
    finally:
        _close_resources(world, camera)
    check = pairing_check(records)
    eligible = check['paired'] and check['context'] is not None
    # No simulation failure is ever presented as a learned negative example.
    targets = {name: (int(records[name]['result'] == 'save')
                     if eligible and records[name]['result'] != 'sim_error' else None)
               for name in DEFENSE_NAMES}
    return {'dataset': DATASET_VERSION, 'split': job['split'], 'style': job['style'],
            'seed': job['seed'], 'episode_index': job['episode_index'],
            'context': check['context'] if eligible else None, 'eligible': bool(eligible),
            'targets': targets, 'outcomes': {name: records[name] for name in DEFENSE_NAMES},
            'pairing': check, 'defense_order': list(order), 'wall_s': time.perf_counter()-start}


def _evaluate_job(job):
    world = SensorDefenseDuelWorld(ball_jitter=.05, start_jitter=(.05, .1))
    camera, records = None, {}
    offset = job['seed'] % len(EVALUATION_MODES)
    order = EVALUATION_MODES[offset:]+EVALUATION_MODES[:offset]
    start = time.perf_counter()
    try:
        for mode in order:
            try:
                camera = _camera(world, camera)
                keeper = (LearnedDefenseKeeper(job['model']) if mode == 'learned'
                          else VisionLibraryKeeper(mode))
                records[mode] = run_camera_episode(world, camera,
                    _striker(job['style'], job['seed'], job['trained']), keeper, job['seed'])
            except (RuntimeError, ValueError, OSError) as error:
                records[mode] = _error_record(job['seed'], error)
                try:
                    _close_resources(world, camera)
                finally:
                    camera = None
    finally:
        _close_resources(world, camera)
    return {'evaluation': EVALUATION_VERSION, 'split': 'test', 'style': job['style'],
            'seed': job['seed'], 'episode_index': job['episode_index'],
            'records': records, 'mode_order': list(order), 'wall_s': time.perf_counter()-start}


def _checkpoint_path(output, split, style, seed):
    return Path(output)/'checkpoints'/split/style/(str(seed)+'.json')


def _checkpoint_read(path, fingerprint, identity):
    if not path.exists():
        return None
    checkpoint = _read_json(path)
    payload = checkpoint.get('payload')
    if (checkpoint.get('version') != TRAINING_VERSION or checkpoint.get('fingerprint') != fingerprint
            or checkpoint.get('identity') != identity or
            checkpoint.get('payload_sha256') != _digest(payload) or not isinstance(payload, dict) or
            any(payload.get(key) != value for key, value in identity.items())):
        raise ValueError('checkpoint identity/hash differs: '+str(path))
    return payload


def _checkpoint_write(path, fingerprint, identity, payload):
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in identity.items()):
        raise ValueError('checkpoint payload identity differs from its job')
    _atomic_json(path, {'version': TRAINING_VERSION, 'fingerprint': fingerprint,
                       'identity': identity, 'payload_sha256': _digest(payload), 'payload': payload})


def _jobs(manifest, split):
    configuration = manifest['configuration']
    trained = {style: player_spec('models/rl/striker_'+style)[1]
               for style in configuration['styles'] if style in ('v1', 'v2')}
    return [{'split': split, 'style': style, 'episode_index': index,
             'seed': episode_seed(split, style, index), 'trained': trained}
            for style in configuration['styles']
            for index in range(configuration['episodes_per_style'][split])]


def _execute_jobs(jobs, function, output, fingerprint, workers=4, progress=print):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('workers must be in [1, 16]')
    rows, pending = [], []
    for job in jobs:
        identity = {key: job[key] for key in ('split', 'style', 'seed', 'episode_index')}
        path = _checkpoint_path(output, job['split'], job['style'], job['seed'])
        cached = _checkpoint_read(path, fingerprint, identity)
        if cached is None:
            pending.append((job, path, identity))
        else:
            rows.append(cached)
    total, start = len(jobs), time.perf_counter()
    if progress:
        progress('%s: %d/%d cases cached; %d workers' % (jobs[0]['split'], len(rows), total, workers), flush=True)

    def finish(payload, path, identity):
        _checkpoint_write(path, fingerprint, identity, payload)
        rows.append(payload)
        if progress:
            progress('[%d/%d] %s / %s / %d (%.1f s)' %
                     (len(rows), total, identity['split'], identity['style'],
                      identity['seed'], time.perf_counter()-start), flush=True)

    if workers == 1:
        for job, path, identity in pending:
            finish(function(job), path, identity)
    elif pending:
        # Native IPC and each simulated episode are bounded. On interruption,
        # pending jobs are cancelled and running jobs finish their finally
        # blocks so all engines/renderers are collected before pool shutdown.
        pool = ProcessPoolExecutor(min(workers, len(pending)))
        futures = {}
        try:
            futures = {pool.submit(function, job): (path, identity)
                       for job, path, identity in pending}
            for future in as_completed(futures):
                finish(future.result(), *futures[future])
        finally:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
    return sorted(rows, key=lambda row: (STYLES.index(row['style']), row['episode_index']))


def dataset_summary(rows):
    outcomes = [record for row in rows for record in row['outcomes'].values()]
    return {'contexts': len(rows), 'eligible_contexts': sum(bool(row['eligible']) for row in rows),
            'paired_contexts': sum(bool(row['pairing']['paired']) for row in rows),
            'pairing_mismatches': sum(not row['pairing']['paired'] for row in rows),
            'no_commit_contexts': sum(row['pairing']['reason'] == 'no_commit' for row in rows),
            'duels': len(outcomes), 'sim_errors': sum(r['result'] == 'sim_error' for r in outcomes),
            'no_commit_duels': sum(bool(r.get('no_commit')) for r in outcomes),
            'saves_by_defense': {name: sum(row['outcomes'][name]['result'] == 'save' for row in rows)
                                for name in DEFENSE_NAMES},
            'training_labels_by_defense': {name: sum(row['targets'][name] is not None for row in rows)
                                          for name in DEFENSE_NAMES},
            'result_counts': dict(Counter(r['result'] for r in outcomes))}


def collect(output, manifest, workers=4, progress=print):
    datasets = {}
    for split in ('train', 'validation'):
        rows = _execute_jobs(_jobs(manifest, split), _collect_job, output,
                             manifest['fingerprint'], workers, progress)
        dataset = {'version': DATASET_VERSION, 'split': split, 'fingerprint': manifest['fingerprint'],
                   'context': CONTEXT_VERSION, 'objective': OBJECTIVE_VERSION,
                   'defenses': list(DEFENSE_NAMES), 'summary': dataset_summary(rows), 'rows': rows}
        _atomic_json(Path(output)/(split+'_dataset.json'), dataset)
        datasets[split] = dataset
    return datasets


def _load_dataset(output, manifest, split):
    path = Path(output)/(split+'_dataset.json')
    dataset = _read_json(path)
    expected = len(manifest['configuration']['styles'])*manifest['configuration']['episodes_per_style'][split]
    if (dataset.get('version') != DATASET_VERSION or dataset.get('fingerprint') != manifest['fingerprint']
            or dataset.get('split') != split or len(dataset.get('rows', ())) != expected):
        raise ValueError('complete matching '+split+' dataset is required')
    # Check reconstructed rows against each independently validated checkpoint.
    for row in dataset['rows']:
        identity = {key: row[key] for key in ('split', 'style', 'seed', 'episode_index')}
        cached = _checkpoint_read(_checkpoint_path(output, split, row['style'], row['seed']),
                                  manifest['fingerprint'], identity)
        if cached != row:
            raise ValueError('dataset differs from checkpoint: '+str(row['seed']))
    return dataset


def fit(output, manifest, model_output, progress=print):
    train = _load_dataset(output, manifest, 'train')
    validation = _load_dataset(output, manifest, 'validation')
    model = fit_model(train['rows'], validation['rows'],
                      epochs=manifest['configuration']['fit_epochs'], seed=SPLIT_BASES['train'])
    provenance = {
        'version': TRAINING_VERSION, 'fingerprint': manifest['fingerprint'],
        'objective': OBJECTIVE_VERSION, 'context': CONTEXT_VERSION,
        'train_dataset_sha256': _sha256(Path(output)/'train_dataset.json'),
        'validation_dataset_sha256': _sha256(Path(output)/'validation_dataset.json'),
        'train_seed_base': SPLIT_BASES['train'], 'validation_seed_base': SPLIT_BASES['validation'],
        'test_seed_base': SPLIT_BASES['test'], 'source': manifest['source'],
        'train_summary': train['summary'], 'validation_summary': validation['summary'],
        'limitations': LIMITATIONS, 'hardware_validated': False,
        'test_used_for_fit': False}
    model = seal_model(model, provenance)
    validate_model(model)
    model_output = Path(model_output)
    if model_output.exists():
        previous = load_model(model_output)
        if previous.get('training_provenance', {}).get('fingerprint') != manifest['fingerprint']:
            raise ValueError('refusing to overwrite a model from another run: '+str(model_output))
    temporary = model_output.with_name(model_output.name+'.tmp-'+str(os.getpid()))
    try:
        save_model(model, temporary)
        os.replace(temporary, model_output)
    finally:
        if temporary.exists():
            temporary.unlink()
    frozen = {'version': TRAINING_VERSION, 'fingerprint': manifest['fingerprint'],
              'model_path': str(model_output.resolve()), 'model_sha256': _sha256(model_output),
              'model': MODEL_VERSION, 'policy': POLICY_VERSION,
              'fit_report': model['fit_report'], 'test_used_for_fit': False}
    _atomic_json(Path(output)/'fit_report.json', frozen)
    if progress:
        progress('Frozen model: '+str(model_output.resolve()), flush=True)
    return frozen


def evaluate(output, manifest, model_output, workers=4, progress=print):
    """Paired final test with immutable weights; this function never fits."""
    frozen = _read_json(Path(output)/'fit_report.json')
    model_output = Path(model_output)
    model_hash = _sha256(model_output)
    if (frozen.get('fingerprint') != manifest['fingerprint'] or
            frozen.get('model_sha256') != model_hash):
        raise ValueError('evaluation requires the exact model frozen in fit_report.json')
    model = load_model(model_output)
    if model.get('training_provenance', {}).get('fingerprint') != manifest['fingerprint']:
        raise ValueError('model training fingerprint differs')
    jobs = _jobs(manifest, 'test')
    for job in jobs:
        job['model'] = model
    # Test checkpoint identity additionally binds the exact trained weights.
    evaluation_fingerprint = _digest({'run': manifest['fingerprint'], 'model': model_hash,
                                      'evaluation': EVALUATION_VERSION})
    start = time.perf_counter()
    rows = _execute_jobs(jobs, _evaluate_job, output, evaluation_fingerprint, workers, progress)
    if _sha256(model_output) != model_hash:
        raise RuntimeError('model changed during the final test')
    groups = {mode: {style: [] for style in manifest['configuration']['styles']}
              for mode in EVALUATION_MODES}
    for row in rows:
        for mode, record in row['records'].items():
            groups[mode][row['style']].append(dict(record, episode_index=row['episode_index']))

    def paired_indices(group):
        # Each style uses independent physical seeds. compare_styles requires
        # common cluster keys across styles, so cluster the equally numbered
        # paired cases while retaining the actual seed in all raw records.
        return {style: [dict(record, simulation_seed=record['seed'], seed=record['episode_index'])
                        for record in records] for style, records in group.items()}

    comparisons = {mode+'_vs_original': compare_styles(paired_indices(groups[mode]),
                                                       paired_indices(groups['original']))
                   for mode in ('learned', 'balanced')}
    comparisons['learned_vs_balanced'] = compare_styles(paired_indices(groups['learned']),
                                                       paired_indices(groups['balanced']))
    for comparison in comparisons.values():
        comparison['ci_method'] = ('paired bootstrap by episode index across independently seeded styles; '
                                   'raw simulation seeds retained in records')
        comparison['independent_cases_per_style'] = manifest['configuration']['episodes_per_style']['test']
    report = {'version': EVALUATION_VERSION, 'fingerprint': manifest['fingerprint'],
              'evaluation_fingerprint': evaluation_fingerprint,
              'model_sha256': model_hash, 'model_path': str(model_output.resolve()),
              'model_frozen_before_test': True, 'test_used_for_fit': False,
              'styles': manifest['configuration']['styles'],
              'episodes_per_style': manifest['configuration']['episodes_per_style']['test'],
              'test_seed_base': SPLIT_BASES['test'], 'camera_pipeline': PIPELINE_VERSION,
              'context': CONTEXT_VERSION, 'action': ACTION_VERSION,
              'summary': {mode: aggregate(group) for mode, group in groups.items()},
              'comparisons': comparisons,
              'defense_counts': {mode: dict(Counter(record.get('commit_defense') or 'none'
                  for records in group.values() for record in records)) for mode, group in groups.items()},
              'no_commit_counts': {mode: sum(bool(record.get('no_commit'))
                  for records in group.values() for record in records) for mode, group in groups.items()},
              'records': groups, 'cases': rows, 'wall_s': time.perf_counter()-start,
              'source': manifest['source'], 'limitations': LIMITATIONS, 'hardware_validated': False}
    _atomic_json(Path(output)/'test_report.json', report)
    return report


def run(output, *, model_output='models/vision/keeper_selector_v1.json', stage='all',
        train_episodes=12, validation_episodes=6, test_episodes=20, styles=STYLES,
        workers=4, fit_epochs=1500, resume=False, progress=print):
    if stage not in ('collect', 'fit', 'evaluate', 'all'):
        raise ValueError('unknown stage')
    manifest = prepare_run(output, train_episodes=train_episodes,
                           validation_episodes=validation_episodes, test_episodes=test_episodes,
                           styles=styles, fit_epochs=fit_epochs, resume=resume)
    result = {'manifest': manifest}
    if stage in ('collect', 'all'):
        result['datasets'] = collect(output, manifest, workers, progress)
    if stage in ('fit', 'all'):
        result['fit'] = fit(output, manifest, model_output, progress)
    if stage in ('evaluate', 'all'):
        result['test'] = evaluate(output, manifest, model_output, workers, progress)
    return result
