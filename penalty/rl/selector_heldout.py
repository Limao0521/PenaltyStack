"""Frozen, balanced blind test for the six-defense selector.

An integration smoke exposed test index zero for two styles. The separately
fixed protocol excludes index zero for every style and uses indices 1..20.
This module reuses the original model-bound test checkpoints for indices
1..19, collects the eight additional cases at index 20, and never fits.
The existing training sources, model weights and exploratory report remain
unchanged. Raw selected rows live in one artifact referenced by SHA-256.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from penalty.keeper_defenses import ACTION_VERSION, DEFENSE_NAMES
from penalty.keeper_selector import CONTEXT_VERSION, load_model
from penalty.motion_contract import ROOT
from penalty.rl.keeper_benchmark import STYLES, aggregate, compare_styles
from penalty.rl.selector_training import (
    EVALUATION_MODES, EVALUATION_VERSION, LIMITATIONS, SPLIT_BASES,
    _atomic_json, _checkpoint_path, _checkpoint_read, _digest, _evaluate_job, _execute_jobs, _jobs,
    _read_json, _sha256, episode_seed, prepare_run,
)
from penalty.sim_orientation import PIPELINE_VERSION


PROTOCOL_VERSION = 'keeper_selector_blind_test_protocol_v1'
HELDOUT_VERSION = 'keeper_selector_blind_evaluation_v1'
RAW_VERSION = 'keeper_selector_blind_cases_v1'
INDICES = tuple(range(1, 21))
EXCLUDED_INDICES = (0,)
EXTRA_SOURCE_PATHS = ('penalty/rl/selector_heldout.py', 'tools/evaluate_keeper_selector_heldout.py')
PROTOCOL_REASON = (
    'The integration smoke used test index 0 for center and mixed. Exclude index 0 for every style, '
    'maintaining equal style weights, and add index 20. Protocol fixed before fitting and before '
    'any production test result.'
)


def validate_protocol(protocol, manifest):
    """Accept only the complete balanced protocol bound to this frozen run."""
    if not isinstance(protocol, dict) or not isinstance(manifest, dict):
        raise ValueError('protocol and run manifest must be objects')
    configuration = manifest['configuration']
    if (protocol.get('version') != PROTOCOL_VERSION or
            protocol.get('run_fingerprint') != manifest.get('fingerprint') or
            protocol.get('styles') != configuration.get('styles') or
            protocol.get('styles') != list(STYLES) or
            type(protocol.get('test_base')) is not int or
            protocol['test_base'] != SPLIT_BASES['test'] or
            configuration.get('split_bases', {}).get('test') != SPLIT_BASES['test'] or
            protocol.get('episode_indices') != list(INDICES) or
            any(type(index) is not int for index in protocol.get('episode_indices', ())) or
            protocol.get('excluded_episode_indices') != list(EXCLUDED_INDICES) or
            any(type(index) is not int for index in protocol.get('excluded_episode_indices', ())) or
            type(protocol.get('episodes_per_style')) is not int or
            protocol['episodes_per_style'] != len(INDICES) or
            configuration.get('episodes_per_style', {}).get('test') != len(INDICES) or
            protocol.get('model_weights_may_be_adjusted_with_test') is not False):
        raise ValueError('blind protocol differs from the fixed run/styles/base/indices 1..20')
    try:
        created = datetime.fromisoformat(protocol['created_utc'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('protocol requires an explicit UTC creation timestamp') from error
    if created.tzinfo is None or created.utcoffset().total_seconds() != 0:
        raise ValueError('protocol creation timestamp must be UTC')
    return protocol


def initialize_protocol(output):
    """Fix the blind test before fitting; never replace an existing protocol."""
    output = Path(output)
    if (output/'fit_report.json').exists():
        raise ValueError('blind protocol must be initialized before fit_report.json exists')
    original = _read_json(output/'run_manifest.json')
    configuration = original['configuration']
    counts = configuration['episodes_per_style']
    manifest = prepare_run(output, train_episodes=counts['train'],
                           validation_episodes=counts['validation'], test_episodes=counts['test'],
                           styles=configuration['styles'], fit_epochs=configuration['fit_epochs'], resume=True)
    path = output/'heldout_protocol.json'
    if path.exists():
        # Preserve the original timestamp, reason and bytes. Scope validation
        # rejects any competing run, style set, indices, base or adjustment.
        return validate_protocol(_read_json(path), manifest)
    protocol = {'version': PROTOCOL_VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'run_fingerprint': manifest['fingerprint'], 'styles': list(STYLES),
                'test_base': SPLIT_BASES['test'], 'episode_indices': list(INDICES),
                'episodes_per_style': len(INDICES), 'excluded_episode_indices': list(EXCLUDED_INDICES),
                'reason': PROTOCOL_REASON, 'model_weights_may_be_adjusted_with_test': False}
    validate_protocol(protocol, manifest)
    _atomic_json(path, protocol)
    return protocol


def heldout_jobs(manifest, protocol, model):
    """Use exactly the original environment/player setup and cache identities."""
    validate_protocol(protocol, manifest)
    templates = {}
    for job in _jobs(manifest, 'test'):
        templates.setdefault(job['style'], job)
    return [dict(templates[style], episode_index=index,
                 seed=episode_seed('test', style, index), model=model)
            for style in protocol['styles'] for index in protocol['episode_indices']]


def select_cases(rows, manifest, protocol):
    """Select exact protocol cases and verify the three peers share each seed."""
    validate_protocol(protocol, manifest)
    selected = {}
    for row in rows:
        key = (row.get('style'), row.get('episode_index'))
        if key[0] not in protocol['styles'] or key[1] not in protocol['episode_indices']:
            continue
        if key in selected:
            raise ValueError('duplicate blind-test case: '+str(key))
        expected = episode_seed('test', key[0], key[1])
        if (row.get('split') != 'test' or row.get('evaluation') != EVALUATION_VERSION or
                row.get('seed') != expected or set(row.get('records', {})) != set(EVALUATION_MODES)):
            raise ValueError('blind case identity/modes differ: '+str(key))
        reset_reference = None
        for mode, record in row['records'].items():
            if record.get('seed') != expected:
                raise ValueError('paired policy has a different physical seed: '+mode)
            # Reset metadata is evaluation-only. It can confirm identical
            # engine seeds/layout without becoming an input to the selector.
            if record.get('result') != 'sim_error' and record.get('reset_info') is not None:
                reset = record['reset_info']
                comparable = {name: reset.get(name) for name in ('engine_seed', 'ball_xy', 'start_offsets')}
                if reset_reference is None:
                    reset_reference = comparable
                elif comparable != reset_reference:
                    raise ValueError('paired policy reset conditions differ: '+mode)
        selected[key] = row
    expected_keys = {(style, index) for style in protocol['styles'] for index in protocol['episode_indices']}
    if set(selected) != expected_keys:
        raise ValueError('complete 160-case blind test is required')
    return [selected[(style, index)] for style in protocol['styles'] for index in protocol['episode_indices']]


def paired_comparison(first, second, samples=10000, seed=0):
    """Equal-style macro gain with independently resampled paired style cases.

    Styles use different physical seeds. The published compare_styles helper
    supplies exact per-style paired tests; independent bootstrap draws within
    each style avoid treating unrelated equal-numbered cases as one cluster.
    """
    if set(first) != set(second) or not first or type(samples) is not int or samples < 1:
        raise ValueError('paired comparison requires matching nonempty styles and positive samples')
    statistics, gains, intervals, bootstraps = {}, [], {}, []
    rng = np.random.default_rng(seed)
    for style in sorted(first):
        result = compare_styles({style: first[style]}, {style: second[style]}, samples=samples)
        statistics[style] = result['per_style'][style]
        intervals[style] = result['macro_gain_ci95']
        a = {record['seed']: record for record in first[style]}
        b = {record['seed']: record for record in second[style]}
        seeds = sorted(a)
        if not seeds:
            raise ValueError('paired styles require at least one case')
        differences = np.asarray([int(a[key]['result'] == 'save')-int(b[key]['result'] == 'save')
                                  for key in seeds], dtype=float)
        gains.append(float(differences.mean()))
        bootstraps.append(rng.choice(differences, (samples, len(differences)), replace=True).mean(axis=1))
    macro_samples = np.mean(bootstraps, axis=0)
    return {'per_style': statistics, 'macro_gain': float(np.mean(gains)),
            'macro_gain_ci95': np.quantile(macro_samples, [.025, .975]).tolist(),
            'per_style_gain_ci95': intervals,
            'ci_method': 'paired bootstrap independently within each style; equal weight per style',
            'independent_cases_per_style': {style: len(records) for style, records in first.items()}}


def summarize_heldout(rows, manifest, protocol, model_hash):
    rows = select_cases(rows, manifest, protocol)
    groups = {mode: {style: [] for style in protocol['styles']} for mode in EVALUATION_MODES}
    for row in rows:
        for mode, record in row['records'].items():
            groups[mode][row['style']].append(dict(record, episode_index=row['episode_index']))

    comparisons = {}
    for first, second in (('learned', 'original'), ('balanced', 'original'), ('learned', 'balanced')):
        comparisons[first+'_vs_'+second] = paired_comparison(groups[first], groups[second])
    summaries = {mode: aggregate(group) for mode, group in groups.items()}
    defense_counts, no_commit_counts = {}, {}
    for mode, group in groups.items():
        records = [record for values in group.values() for record in values]
        counts = Counter(record.get('commit_defense') or 'none' for record in records)
        defense_counts[mode] = {name: counts[name] for name in DEFENSE_NAMES}
        defense_counts[mode]['none'] = counts['none']
        no_commit_counts[mode] = sum(bool(record.get('no_commit', record.get('commit_time') is None))
                                     for record in records)
    return {'version': HELDOUT_VERSION, 'fingerprint': manifest['fingerprint'],
            'model_sha256': model_hash, 'model_frozen_before_test': True,
            'test_used_for_fit': False, 'styles': protocol['styles'],
            'episodes_per_style': len(INDICES), 'episode_indices': list(INDICES),
            'excluded_episode_indices': list(EXCLUDED_INDICES),
            'test_seed_base': SPLIT_BASES['test'], 'selected_cases': len(rows),
            'policy_duels': len(rows)*len(EVALUATION_MODES), 'camera_pipeline': PIPELINE_VERSION,
            'context': CONTEXT_VERSION, 'action': ACTION_VERSION,
            'summary': summaries, 'comparisons': comparisons, 'defense_counts': defense_counts,
            'no_commit_counts': no_commit_counts,
            'sim_error_counts': {mode: summary['sim_errors'] for mode, summary in summaries.items()},
            'hardware_validated': False, 'limitations': LIMITATIONS}


def _validate_pre_fit_time(protocol, protocol_path, model_path, fit_report_path):
    """Check local provenance supporting the pre-fit protocol declaration.

    The existing fit format has no embedded timestamp, so this explicitly
    uses the UTC declaration and unchanged local artifact modification times.
    It does not claim externally authenticated timestamps.
    """
    created = datetime.fromisoformat(protocol['created_utc']).timestamp()
    protocol_mtime = Path(protocol_path).stat().st_mtime
    model_mtime = Path(model_path).stat().st_mtime
    fit_mtime = Path(fit_report_path).stat().st_mtime
    if max(created, protocol_mtime) > min(model_mtime, fit_mtime):
        raise ValueError('blind protocol must have been fixed before model fitting')
    return {'protocol_created_utc': protocol['created_utc'],
            'protocol_file_modified_utc': datetime.fromtimestamp(protocol_mtime, timezone.utc).isoformat(),
            'model_file_modified_utc': datetime.fromtimestamp(model_mtime, timezone.utc).isoformat(),
            'fit_report_file_modified_utc': datetime.fromtimestamp(fit_mtime, timezone.utc).isoformat(),
            'protocol_precedes_fit': True,
            'method': 'declared UTC creation time and local file modification times; no external timestamp authority'}


def evaluate_heldout(output, model_output, workers=6, progress=print):
    """Resume the fixed 160-case test without training or changing old reports."""
    output, model_output = Path(output), Path(model_output)
    original_manifest = _read_json(output/'run_manifest.json')
    configuration = original_manifest['configuration']
    counts = configuration['episodes_per_style']
    manifest = prepare_run(output, train_episodes=counts['train'],
                           validation_episodes=counts['validation'], test_episodes=counts['test'],
                           styles=configuration['styles'], fit_epochs=configuration['fit_epochs'], resume=True)
    protocol_path = output/'heldout_protocol.json'
    protocol = validate_protocol(_read_json(protocol_path), manifest)
    protocol_hash = _sha256(protocol_path)
    fit_path = output/'fit_report.json'
    fit_report = _read_json(fit_path)
    fit_hash = _sha256(fit_path)
    model_hash = _sha256(model_output)
    if (fit_report.get('fingerprint') != manifest['fingerprint'] or
            fit_report.get('model_sha256') != model_hash or
            fit_report.get('test_used_for_fit') is not False):
        raise ValueError('held-out test requires the exact model frozen in fit_report.json')
    model = load_model(model_output)
    if model.get('training_provenance', {}).get('fingerprint') != manifest['fingerprint']:
        raise ValueError('model training fingerprint differs from the original run')
    pre_fit = _validate_pre_fit_time(protocol, protocol_path, model_output, fit_path)
    evaluation_fingerprint = _digest({'run': manifest['fingerprint'], 'model': model_hash,
                                      'evaluation': EVALUATION_VERSION})
    exploratory_path = output/'test_report.json'
    exploratory_hash = _sha256(exploratory_path) if exploratory_path.exists() else None
    if exploratory_hash is not None:
        exploratory = _read_json(exploratory_path)
        if (exploratory.get('fingerprint') != manifest['fingerprint'] or
                exploratory.get('evaluation_fingerprint') != evaluation_fingerprint or
                exploratory.get('model_sha256') != model_hash):
            raise ValueError('exploratory report does not bind the same frozen model/run')
    extra_hashes = {name: _sha256(ROOT/name) for name in EXTRA_SOURCE_PATHS}
    start = time.perf_counter()
    jobs = heldout_jobs(manifest, protocol, model)
    cached_cases = 0
    for job in jobs:
        identity = {key: job[key] for key in ('split', 'style', 'seed', 'episode_index')}
        path = _checkpoint_path(output, 'test', job['style'], job['seed'])
        cached_cases += _checkpoint_read(path, evaluation_fingerprint, identity) is not None
    rows = _execute_jobs(jobs, _evaluate_job, output, evaluation_fingerprint, workers, progress)
    rows = select_cases(rows, manifest, protocol)
    # No mutation of any bound artifact is permitted while the final test runs.
    unchanged = (_sha256(model_output) == model_hash and _sha256(protocol_path) == protocol_hash
                 and _sha256(fit_path) == fit_hash and
                 all(_sha256(ROOT/name) == expected for name, expected in extra_hashes.items()) and
                 all(_sha256(ROOT/name) == expected
                     for name, expected in configuration['source_hashes'].items()))
    if exploratory_hash is not None:
        unchanged &= exploratory_path.exists() and _sha256(exploratory_path) == exploratory_hash
    elif exploratory_path.exists():
        unchanged = False
    if not unchanged:
        raise RuntimeError('frozen model/protocol/fit/source/exploratory artifact changed during the test')
    raw_path = output/'selected_case_rows.json'
    _atomic_json(raw_path, {'version': RAW_VERSION, 'fingerprint': manifest['fingerprint'],
                           'evaluation_fingerprint': evaluation_fingerprint,
                           'protocol_sha256': protocol_hash, 'model_sha256': model_hash,
                           'rows': rows})
    report = summarize_heldout(rows, manifest, protocol, model_hash)
    report.update({'evaluation_fingerprint': evaluation_fingerprint,
                   'model_path': str(model_output.resolve()),
                   'protocol': {'path': protocol_path.name, 'sha256': protocol_hash},
                   'fit_report': {'path': fit_path.name, 'sha256': fit_hash},
                   'selected_raw_rows': {'path': raw_path.name, 'sha256': _sha256(raw_path),
                                         'cases': len(rows), 'duels': len(rows)*len(EVALUATION_MODES)},
                   'exploratory_report': {'path': exploratory_path.name, 'sha256': exploratory_hash,
                                          'unchanged': True, 'indices': list(range(20))},
                   'pre_fit_protocol_provenance': pre_fit,
                   'checkpoint_identity': 'original evaluation fingerprint and physical split/style/seed/index',
                   'selection': 'indices 1..20 only; existing original model-bound caches are reused',
                   'cached_cases_before_run': cached_cases,
                   'collected_cases_this_run': len(rows)-cached_cases,
                   'source': manifest['source'], 'extra_source_hashes': extra_hashes,
                   'wall_s': time.perf_counter()-start})
    _atomic_json(output/'heldout_report.json', report)
    return report
