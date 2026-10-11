"""Bounded motor-skill search in free-root B-Human physics and RGB perception.

The published vision v2 predictor and balanced defense selection stay fixed.
This experiment changes real upstream keyframe durations and joint targets,
executed by the original B-Human phase, interpolation and motor controller.
Training/validation/blind-test seeds are distinct and fixed before search.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time

import numpy as np

from penalty.keeper_defenses import DEFENSE_NAMES
from penalty.keeper_library import VisionLibraryKeeper
from penalty.motion_contract import ROOT
from penalty.rl.duel import player_spec
from penalty.rl.keeper_benchmark import STYLES, aggregate
from penalty.rl.selector_heldout import paired_comparison
from penalty.rl.selector_training import (
    _atomic_json, _digest, _read_json, _sha256, _striker, _camera,
    _close_resources, run_camera_episode, source_metadata,
)
from penalty.rl.motion_world import MotionDuelWorld
from penalty.skills.keeper_motion import MotionParameters, ParameterizedKeeperMotion


TRAINING_VERSION = 'keeper_motion_training_v1'
MODEL_VERSION = 'keeper_motion_profile_model_v1'
OBJECTIVE_VERSION = 'keeper_motion_outcome_v2'
SPLIT_BASES = {'train': 81001, 'validation': 82001, 'test': 83001}
SMOKE_BASE = 80001
NEW_SOURCES = (
    'penalty/rl/motion_training.py', 'penalty/rl/motion_world.py',
    'penalty/skills/keeper_motion.py', 'tools/train_keeper_motion.py',
    'tools/validate_keeper_motion.py', 'tools/watch_keeper_motion.py',
    'native/bhuman_motion/main.cpp', 'native/bhuman_motion/factories.h',
    'native/bhuman_motion/keyframe_factory.cpp', 'tools/build_motion_runtime.py',
    'penalty/rl/selector_heldout.py',
)
LIMITATIONS = (
    'MuJoCo simulation only; no physical NAO calibration. Own simulated sensors '
    'are ideal and the physical clock pauses during RGB inference. The keeper '
    'uses camera-derived ball measurements; the striker has privileged scene '
    'observations with PerceptionNoise v1. v1/v2 opponents are our trained '
    'strikers, not an autonomous B-Human team. Motion parameters are shared '
    'across mirrored defenses. Balanced normally selects four of six guards; '
    'all six are checked for execution/recovery, not independently trained '
    'or statistically ranked. Parameter search is a pilot, not full-joint PPO.'
)


def episode_seed(split, style, index):
    if (split not in SPLIT_BASES or style not in STYLES or type(index) is not int
            or not 0 <= index < 10000):
        raise ValueError('invalid split/style/episode index')
    return SPLIT_BASES[split]*1_000_000 + STYLES.index(style)*10000 + index


def candidates(count=16, seed=81001):
    """Nominal control, four single-parameter probes, then Latin-hypercube search."""
    if type(count) is not int or not 5 <= count <= 64:
        raise ValueError('candidate count must be 5..64')
    skill = ParameterizedKeeperMotion()
    nominal = asdict(MotionParameters())
    profiles = [nominal]
    for name, value in (('guard_time_scale', .85), ('jump_time_scale', .85),
                        ('arm_spread_rad', .10), ('guard_knee_offset_rad', .08)):
        profiles.append(dict(nominal, **{name: value}))
    random_count = count-len(profiles)
    rng = np.random.default_rng(seed)
    if random_count:
        bounds = skill.bounds()
        columns = {name: (rng.permutation(random_count)+rng.uniform(size=random_count))/random_count
                   for name in bounds}
        for index in range(random_count):
            profiles.append({name: round(low + columns[name][index]*(high-low), 8)
                             for name, (low, high) in bounds.items()})
    result = []
    for index, params in enumerate(profiles):
        ParameterizedKeeperMotion.validate(MotionParameters(**params))
        result.append({'id': 'nominal' if not index else 'candidate_%02d' % index,
                       'parameters': params})
    return result


def _sources():
    sources = source_metadata()
    sources['hashes'].update({name: _sha256(ROOT/name) for name in NEW_SOURCES})
    return sources


def prepare_run(output, count=16, train_episodes=4, validation_episodes=8,
                test_episodes=20, resume=False):
    output = Path(output)
    counts = dict(train=train_episodes, validation=validation_episodes, test=test_episodes)
    if any(type(value) is not int or not 1 <= value < 10000 for value in counts.values()):
        raise ValueError('positive episode counts required')
    sources = _sources()
    configuration = {
        'training': TRAINING_VERSION, 'objective': OBJECTIVE_VERSION,
        'model': MODEL_VERSION, 'skill': 'keeper_motion_skill_v1',
        'base_policy': 'vision_keeper_v2', 'selection_mode': 'balanced',
        'styles': list(STYLES), 'split_bases': SPLIT_BASES,
        'episode_indices': {split: list(range(value)) for split, value in counts.items()},
        'episodes_per_style': counts, 'candidates': candidates(count),
        'shortlist': 'two highest-scoring safe nonnominal train profiles plus nominal',
        'selection': 'validation macro saves, worst style, least parameter deviation',
        'test_modes': ['trained', 'balanced'], 'test_may_adjust_parameters': False,
        'objective_definition': 'equal-style save rate; all outcomes/no-commit in denominator; '
                                'simulation errors abort selection; execution/recovery is a prerequisite',
        'world': {'ball_jitter': .05, 'start_jitter': [.05, .1]},
        'safety_seed': 80002, 'safety_defenses': list(DEFENSE_NAMES),
        'source_hashes': sources['hashes'], 'runtime_versions': sources['versions'],
    }
    fingerprint = _digest(configuration)
    path = output/'run_manifest.json'
    if path.exists():
        existing = _read_json(path)
        if (not resume or existing.get('fingerprint') != fingerprint
                or existing.get('configuration') != configuration
                or _digest(existing.get('configuration')) != fingerprint):
            raise ValueError('existing run requires --resume and identical configuration/sources/runtime')
        return existing
    if resume:
        raise ValueError('--resume requires an existing run')
    if output.exists() and any(output.iterdir()):
        raise ValueError('new output directory must be empty')
    manifest = {'version': TRAINING_VERSION, 'fingerprint': fingerprint,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'configuration': configuration, 'source': sources,
                'test_protocol_fixed_before_training': True,
                'hardware_validated': False, 'limitations': LIMITATIONS}
    _atomic_json(path, manifest)
    return manifest


def check_sources(manifest):
    if any(_sha256(ROOT/name) != expected
           for name, expected in manifest['configuration']['source_hashes'].items()):
        raise RuntimeError('run sources/model/native runtime changed')


def _checkpoint(output, identity, fingerprint, payload=None):
    path = Path(output)/'checkpoints'/identity['split']/identity['style']/(str(identity['seed'])+'.json')
    if payload is not None:
        if any(payload.get(name) != value for name, value in identity.items()):
            raise ValueError('job/payload identity differs')
        _atomic_json(path, {'version': TRAINING_VERSION, 'fingerprint': fingerprint,
                           'identity': identity, 'payload_sha256': _digest(payload), 'payload': payload})
        return payload
    if not path.exists():
        return None
    value = _read_json(path)
    payload = value.get('payload')
    if (value.get('version') != TRAINING_VERSION or value.get('fingerprint') != fingerprint
            or value.get('identity') != identity or
            value.get('payload_sha256') != _digest(payload) or not isinstance(payload, dict)
            or any(payload.get(name) != expected for name, expected in identity.items())):
        raise ValueError('invalid checkpoint binding: '+str(path))
    return value['payload']


def _execute(jobs, function, output, fingerprint, workers, progress=print):
    if type(workers) is not int or not 1 <= workers <= 16 or not jobs:
        raise ValueError('nonempty jobs and workers 1..16 required')
    rows, pending = [], []
    for job in jobs:
        identity = {key: job[key] for key in ('split', 'style', 'seed', 'episode_index')}
        cached = _checkpoint(output, identity, fingerprint)
        if cached is None:
            pending.append((job, identity))
        else:
            rows.append(cached)
    total, start = len(jobs), time.perf_counter()
    if progress:
        progress('%s: %d/%d cached, %d workers' % (jobs[0]['split'], len(rows), total, workers), flush=True)

    def finish(payload, identity):
        rows.append(_checkpoint(output, identity, fingerprint, payload))
        if progress:
            progress('[%d/%d] %s / %s / %d (%.1f s)' %
                     (len(rows), total, identity['split'], identity['style'], identity['seed'],
                      time.perf_counter()-start), flush=True)
    if workers == 1:
        for job, identity in pending:
            finish(function(job), identity)
    elif pending:
        pool = ProcessPoolExecutor(min(workers, len(pending)))
        futures = {}
        try:
            futures = {pool.submit(function, job): identity for job, identity in pending}
            for future in as_completed(futures):
                finish(future.result(), futures[future])
        finally:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
    return sorted(rows, key=lambda row: (STYLES.index(row['style']) if row['style'] in STYLES else 0,
                                         row['episode_index']))


def _safety_job(job):
    from tools.validate_keeper_motion import validate_motion_case
    cases = []
    for defense in DEFENSE_NAMES:
        summary, trace = validate_motion_case(MotionParameters(**job['profile']['parameters']),
                                              defense, seed=job['seed'])
        path = Path(job['output'])/'safety_traces'/job['profile']['id']/(defense+'.jsonl.gz')
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ('\n'.join(json.dumps(row, separators=(',', ':'), allow_nan=False)
                         for row in trace)+'\n').encode('utf-8')
        path.write_bytes(gzip.compress(raw, mtime=0))
        summary.update(trace=path.relative_to(job['output']).as_posix(),
                       trace_sha256=_sha256(path), trace_frames=len(trace))
        cases.append(summary)
    return {key: job[key] for key in ('split', 'style', 'seed', 'episode_index', 'profile')} | {
        'cases': cases, 'passed': all(case['passed'] for case in cases)}


def safety_screen(output, manifest, workers=6, progress=print):
    configuration = manifest['configuration']
    jobs = [{'split': 'safety', 'style': 'all_defenses',
             'seed': configuration['safety_seed']*1000+index, 'episode_index': index,
             'profile': profile, 'output': str(Path(output).resolve())}
            for index, profile in enumerate(configuration['candidates'])]
    rows = _execute(jobs, _safety_job, output, manifest['fingerprint'], workers, progress)
    if not rows[0]['passed']:
        raise RuntimeError('nominal motion fails execution/recovery gate')
    report = {'version': TRAINING_VERSION, 'fingerprint': manifest['fingerprint'],
              'profiles': rows, 'eligible_profiles': [row['profile']['id'] for row in rows if row['passed']],
              'profile_count': len(rows), 'physical_cases': len(rows)*len(DEFENSE_NAMES),
              'hardware_validated': False, 'single_pose_only': True}
    check_sources(manifest)
    _atomic_json(Path(output)/'safety_report.json', report)
    return report


def _duel_job(job):
    world = MotionDuelWorld(ball_jitter=.05, start_jitter=(.05, .1))
    camera, records = None, {}
    profiles = job['profiles']
    offset = job['seed'] % len(profiles)
    order = profiles[offset:]+profiles[:offset]
    start = time.perf_counter()
    try:
        for profile in order:
            world.set_profile(MotionParameters(**profile['parameters']))
            camera = _camera(world, camera)
            records[profile['id']] = run_camera_episode(
                world, camera, _striker(job['style'], job['seed'], job['trained']),
                VisionLibraryKeeper('balanced'), job['seed'])
            record = records[profile['id']]
            if record['result'] == 'sim_error':
                raise RuntimeError('motion duel failed: '+str(record.get('sim_error')))
    finally:
        _close_resources(world, camera)
    # Existing pure check compares ball-only context, trigger and reset prefix;
    # use a dedicated new-version wrapper because the defenses are not forced.
    paired = motion_pairing(records)
    if not paired['paired']:
        raise RuntimeError('pre-motion paired context/trigger differs: '+str(paired))
    return {key: job[key] for key in ('split', 'style', 'seed', 'episode_index')} | {
        'records': records, 'pairing': paired,
        'profile_order': [profile['id'] for profile in order], 'wall_s': time.perf_counter()-start}


def motion_pairing(records):
    if not records or any(row.get('result') == 'sim_error' for row in records.values()):
        return {'paired': False, 'reason': 'simulation_error_or_empty'}
    first = next(iter(records.values()))
    fields = ('selector_context', 'first_trigger', 'requested_defense', 'commit_defense',
              'commit_time', 'no_commit', 'seed')
    for row in records.values():
        for name in fields:
            if row.get(name) != first.get(name):
                return {'paired': False, 'reason': name+'_differs'}
        for name in ('engine_seed', 'ball_xy', 'start_offsets'):
            if row.get('reset_info', {}).get(name) != first.get('reset_info', {}).get(name):
                return {'paired': False, 'reason': 'reset_'+name+'_differs'}
    return {'paired': True, 'reason': 'identical_pre_motion_prefix',
            'no_commit': first.get('no_commit', first.get('commit_time') is None)}


def _jobs(manifest, split, profiles):
    configuration = manifest['configuration']
    trained = {style: player_spec('models/rl/striker_'+style)[1] for style in ('v1', 'v2')}
    return [{'split': split, 'style': style, 'episode_index': index,
             'seed': episode_seed(split, style, index), 'profiles': profiles, 'trained': trained}
            for style in configuration['styles'] for index in configuration['episode_indices'][split]]


def summarize(rows, profiles):
    groups = {profile['id']: {style: [] for style in STYLES} for profile in profiles}
    for row in rows:
        if set(row['records']) != set(groups) or not row['pairing']['paired']:
            raise ValueError('incomplete/unpaired profile group')
        for name, record in row['records'].items():
            groups[name][row['style']].append(record)
    return {'summary': {name: aggregate(group) for name, group in groups.items()},
            'save_counts': {name: sum(record['result'] == 'save' for records in group.values()
                                      for record in records) for name, group in groups.items()},
            'defense_counts': {name: dict(Counter(record.get('commit_defense') or 'none'
                                                  for records in group.values() for record in records))
                               for name, group in groups.items()},
            'no_commit_counts': {name: sum(bool(record.get('no_commit'))
                                           for records in group.values() for record in records)
                                 for name, group in groups.items()},
            'paired_cases': len(rows), 'duels': sum(len(row['records']) for row in rows)}, groups


def _distance(profile):
    nominal, bounds = asdict(MotionParameters()), ParameterizedKeeperMotion().bounds()
    return sum(((profile['parameters'][name]-nominal[name])/(high-low))**2
               for name, (low, high) in bounds.items())


def rank_profiles(profiles, report):
    """A fixed deterministic rule; ties prefer the unmodified motor skill."""
    if any(report['summary'][profile['id']]['sim_errors'] for profile in profiles):
        raise ValueError('simulation errors invalidate model selection')
    return sorted(profiles, key=lambda profile: (
        report['summary'][profile['id']]['macro_save_rate'],
        report['summary'][profile['id']]['worst_style_save_rate'],
        -_distance(profile), profile['id'] == 'nominal'), reverse=True)


def search(output, manifest, workers=6, progress=print):
    output = Path(output)
    screen = _read_json(output/'safety_report.json')
    if screen.get('fingerprint') != manifest['fingerprint']:
        raise ValueError('safety screen belongs to another run')
    profiles = [profile for profile in manifest['configuration']['candidates']
                if profile['id'] in screen['eligible_profiles']]
    rows = _execute(_jobs(manifest, 'train', profiles), _duel_job, output,
                    _digest({'run': manifest['fingerprint'], 'profiles': profiles}), workers, progress)
    summary, _ = summarize(rows, profiles)
    ranking = rank_profiles(profiles, summary)
    shortlist = [profile for profile in ranking if profile['id'] != 'nominal'][:2]
    shortlist.append(next(profile for profile in profiles if profile['id'] == 'nominal'))
    report = {'version': TRAINING_VERSION, 'split': 'train', 'fingerprint': manifest['fingerprint'],
              'profiles': profiles, 'ranking': [profile['id'] for profile in ranking],
              'shortlist': shortlist, 'objective': OBJECTIVE_VERSION,
              **summary, 'rows': rows}
    check_sources(manifest)
    _atomic_json(output/'train_report.json', report)
    return report


def validate_and_fit(output, manifest, model_output, workers=6, progress=print):
    output, model_output = Path(output), Path(model_output)
    train = _read_json(output/'train_report.json')
    if train.get('fingerprint') != manifest['fingerprint']:
        raise ValueError('training belongs to another run')
    profiles = train['shortlist']
    rows = _execute(_jobs(manifest, 'validation', profiles), _duel_job, output,
                    _digest({'run': manifest['fingerprint'], 'profiles': profiles}), workers, progress)
    summary, groups = summarize(rows, profiles)
    selected = rank_profiles(profiles, summary)[0]
    report = {'version': TRAINING_VERSION, 'split': 'validation',
              'fingerprint': manifest['fingerprint'], 'profiles': profiles,
              'selected_profile': selected, **summary, 'rows': rows,
              'comparisons': {profile['id']+'_vs_nominal': paired_comparison(
                  groups[profile['id']], groups['nominal'])
                  for profile in profiles if profile['id'] != 'nominal'}}
    model = {'version': MODEL_VERSION, 'skill': 'keeper_motion_skill_v1',
             'parameters': selected['parameters'], 'selected_candidate': selected['id'],
             'base_policy': 'vision_keeper_v2', 'selection_mode': 'balanced',
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'training_fingerprint': manifest['fingerprint'],
             'test_used_for_fit': False, 'simulation_only': True,
             'train_report_sha256': _sha256(output/'train_report.json'),
             'limitations': LIMITATIONS}
    model['integrity_sha256'] = _digest(model)
    if model_output.exists():
        existing = load_profile(model_output)
        if (existing['training_fingerprint'] != manifest['fingerprint'] or
                existing['parameters'] != model['parameters'] or
                existing['selected_candidate'] != model['selected_candidate']):
            raise ValueError('existing frozen profile differs; choose a new model path')
        model = existing
    else:
        _atomic_json(model_output, model)
    report.update({'model_sha256': _sha256(model_output), 'test_used_for_fit': False})
    check_sources(manifest)
    _atomic_json(output/'validation_report.json', report)
    fit_path = output/'fit_report.json'
    fit = {'version': TRAINING_VERSION, 'fingerprint': manifest['fingerprint'],
           'model_sha256': _sha256(model_output), 'selected_profile': selected,
           'model_frozen_utc': model['created_utc'], 'test_used_for_fit': False,
           'validation_report_sha256': _sha256(output/'validation_report.json')}
    if fit_path.exists() and _read_json(fit_path) != fit:
        raise ValueError('existing frozen fit differs')
    if not fit_path.exists():
        _atomic_json(fit_path, fit)
    return fit


def load_profile(path):
    model = _read_json(path)
    if (model.get('version') != MODEL_VERSION or model.get('skill') != 'keeper_motion_skill_v1'
            or model.get('test_used_for_fit') is not False or model.get('simulation_only') is not True):
        raise ValueError('invalid motion model version/scope')
    integrity = model.pop('integrity_sha256', None)
    if integrity != _digest(model):
        raise ValueError('motion profile integrity differs')
    model['integrity_sha256'] = integrity
    if (not isinstance(model.get('parameters'), dict) or
            set(model['parameters']) != set(ParameterizedKeeperMotion.RUNTIME_SUPPORTED)
            or model.get('base_policy') != 'vision_keeper_v2'
            or model.get('selection_mode') != 'balanced'):
        raise ValueError('motion model parameters/policy differ from the published contract')
    ParameterizedKeeperMotion.validate(MotionParameters(**model['parameters']))
    return model


def blind_test(output, manifest, model_output, workers=6, progress=print):
    output, model_output = Path(output), Path(model_output)
    model = load_profile(model_output)
    fit_path = output/'fit_report.json'
    fit = _read_json(fit_path)
    model_hash, fit_hash = _sha256(model_output), _sha256(fit_path)
    if (model.get('training_fingerprint') != manifest['fingerprint'] or
            fit.get('fingerprint') != manifest['fingerprint'] or fit.get('model_sha256') != model_hash
            or fit.get('test_used_for_fit') is not False or
            fit.get('validation_report_sha256') != _sha256(output/'validation_report.json')
            or model.get('train_report_sha256') != _sha256(output/'train_report.json')
            or fit.get('selected_profile', {}).get('parameters') != model['parameters']
            or fit.get('selected_profile', {}).get('id') != model['selected_candidate']
            or datetime.fromisoformat(manifest['created_utc']) > datetime.fromisoformat(model['created_utc'])):
        raise ValueError('blind test requires the frozen validation-selected model')
    profiles = [{'id': 'trained', 'parameters': model['parameters']},
                {'id': 'balanced', 'parameters': asdict(MotionParameters())}]
    rows = _execute(_jobs(manifest, 'test', profiles), _duel_job, output,
                    _digest({'run': manifest['fingerprint'], 'model': model_hash}), workers, progress)
    summary, groups = summarize(rows, profiles)
    report = {'version': TRAINING_VERSION, 'split': 'test', 'fingerprint': manifest['fingerprint'],
              'profiles': profiles, 'model_sha256': model_hash,
              'model_frozen_before_test': True, 'test_used_for_fit': False,
              'test_protocol_created_utc': manifest['created_utc'],
              'model_frozen_utc': model['created_utc'], **summary, 'rows': rows,
              'comparisons': {'trained_vs_balanced': paired_comparison(
                  groups['trained'], groups['balanced'])}, 'hardware_validated': False,
              'limitations': LIMITATIONS}
    check_sources(manifest)
    if _sha256(model_output) != model_hash or _sha256(fit_path) != fit_hash:
        raise RuntimeError('frozen model/fit changed during blind test')
    _atomic_json(output/'test_report.json', report)
    return report
