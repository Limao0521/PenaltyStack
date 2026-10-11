"""Blind-test selection and frozen-cache checks without native simulations."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from penalty.keeper_defenses import DEFENSE_NAMES
from penalty.keeper_selector import CONTEXT_VERSION, fit_model, save_model, seal_model
from penalty.rl import selector_heldout as heldout
from penalty.rl import selector_training as training
from penalty.rl.keeper_benchmark import aggregate


def manifest():
    return {'fingerprint': 'frozen-run', 'configuration': {
        'styles': list(training.STYLES), 'episodes_per_style': {'train': 12, 'validation': 6, 'test': 20},
        'split_bases': dict(training.SPLIT_BASES), 'fit_epochs': 3}}


def protocol(run):
    return {'version': heldout.PROTOCOL_VERSION, 'run_fingerprint': run['fingerprint'],
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'styles': list(training.STYLES), 'test_base': training.SPLIT_BASES['test'],
            'episode_indices': list(range(1, 21)), 'excluded_episode_indices': [0],
            'episodes_per_style': 20, 'model_weights_may_be_adjusted_with_test': False}


def templates(run, split):
    return [{'split': split, 'style': style, 'episode_index': 0,
             'seed': training.episode_seed(split, style, 0), 'trained': {}}
            for style in run['configuration']['styles']]


def case(style, index):
    seed = training.episode_seed('test', style, index)
    reset = {'engine_seed': seed, 'ball_xy': [1., 0.], 'start_offsets': [.01, -.01]}
    records = {}
    for mode in training.EVALUATION_MODES:
        result = ('save' if index == 0 else 'timeout' if index == 2 else
                  'sim_error' if index == 1 and mode == 'learned' else
                  'save' if index % {'learned': 3, 'original': 2, 'balanced': 4}[mode] == 0 else 'goal')
        committed = result not in ('timeout', 'sim_error')
        records[mode] = {'seed': seed, 'result': result, 'no_commit': not committed,
                         'commit_time': .2 if committed else None,
                         'commit_defense': 'center' if committed else None,
                         'early_dive': False, 'commit_side': 0, 'reset_info': copy.deepcopy(reset)}
    return {'evaluation': training.EVALUATION_VERSION, 'split': 'test', 'style': style,
            'seed': seed, 'episode_index': index, 'records': records}


def rows():
    return [case(style, index) for style in training.STYLES for index in range(1, 21)]


class HeldoutProtocolTests(unittest.TestCase):
    def test_protocol_requires_exact_pre_fixed_balanced_indices_one_through_twenty(self):
        run = manifest()
        fixed = protocol(run)
        self.assertEqual(heldout.validate_protocol(fixed, run), fixed)
        changes = [('version', 'other'), ('run_fingerprint', 'other-run'),
                   ('styles', list(reversed(training.STYLES))), ('test_base', 76002),
                   ('episode_indices', list(range(20))), ('episode_indices', [1, 1]+list(range(3, 21))),
                   ('episode_indices', [True]+list(range(2, 21))),
                   ('excluded_episode_indices', []), ('episodes_per_style', 19),
                   ('model_weights_may_be_adjusted_with_test', True),
                   ('created_utc', '2026-10-10T12:00:00'),
                   ('created_utc', '2026-10-10T12:00:00+01:00')]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(fixed)
                changed[key] = value
                with self.assertRaises(ValueError):
                    heldout.validate_protocol(changed, run)
        for key, value in (('test', 76002),):
            changed = copy.deepcopy(run)
            changed['configuration']['split_bases'][key] = value
            with self.assertRaises(ValueError):
                heldout.validate_protocol(fixed, changed)

    def test_jobs_match_original_seed_scheme_and_exclude_smoke_index_zero(self):
        run = manifest()
        fixed = protocol(run)
        model = {'frozen': 'weights'}
        with patch.object(heldout, '_jobs', side_effect=templates):
            jobs = heldout.heldout_jobs(run, fixed, model)
        self.assertEqual(len(jobs), 160)
        self.assertEqual({job['episode_index'] for job in jobs}, set(range(1, 21)))
        self.assertEqual(len({job['seed'] for job in jobs}), 160)
        self.assertTrue(all(job['model'] is model for job in jobs))
        for style in training.STYLES:
            selected = [job for job in jobs if job['style'] == style]
            self.assertEqual([job['episode_index'] for job in selected], list(range(1, 21)))
            self.assertEqual([job['seed'] for job in selected],
                             [training.episode_seed('test', style, i) for i in range(1, 21)])

    def test_selection_ignores_exposed_zero_even_if_its_records_are_invalid(self):
        run = manifest()
        fixed = protocol(run)
        exposed = [dict(case(style, 0), records={'exposed': 'not a blind result'})
                   for style in training.STYLES]
        selected = heldout.select_cases(list(reversed(rows()))+exposed, run, fixed)
        self.assertEqual(len(selected), 160)
        self.assertEqual(selected[0]['style'], training.STYLES[0])
        self.assertEqual(selected[0]['episode_index'], 1)
        self.assertTrue(all(row['episode_index'] > 0 for row in selected))

    def test_missing_duplicate_wrong_seed_mode_and_reset_cases_are_rejected(self):
        run = manifest()
        fixed = protocol(run)
        invalid = []
        invalid.append(rows()[:-1])
        invalid.append(rows()+[case('center', 1)])
        altered = rows()
        altered[0]['seed'] += 1
        invalid.append(altered)
        altered = rows()
        altered[0]['records']['original']['seed'] += 1
        invalid.append(altered)
        altered = rows()
        del altered[0]['records']['balanced']
        invalid.append(altered)
        altered = rows()
        altered[3]['records']['original']['reset_info']['ball_xy'][0] += .1
        invalid.append(altered)
        for index, data in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    heldout.select_cases(data, run, fixed)


class HeldoutStatisticsTests(unittest.TestCase):
    def test_aggregate_is_unchanged_and_errors_and_no_commit_stay_in_denominator(self):
        run = manifest()
        fixed = protocol(run)
        data = rows()
        report = heldout.summarize_heldout(data, run, fixed, 'frozen-model-sha')
        for mode, expected in (('learned', .3), ('original', .45), ('balanced', .25)):
            grouped = {style: [row['records'][mode] for row in data if row['style'] == style]
                       for style in training.STYLES}
            self.assertEqual(report['summary'][mode], aggregate(grouped))
            self.assertAlmostEqual(report['summary'][mode]['macro_save_rate'], expected)
            self.assertEqual(sum(g['episodes'] for g in report['summary'][mode]['per_style'].values()), 160)
            self.assertEqual(sum(report['defense_counts'][mode].values()), 160)
        self.assertEqual(report['sim_error_counts']['learned'], 8)
        self.assertEqual(report['no_commit_counts']['learned'], 16)
        self.assertEqual(report['defense_counts']['learned']['none'], 16)
        self.assertEqual(report['no_commit_counts']['original'], 8)
        self.assertEqual(report['policy_duels'], 480)
        self.assertEqual(report['episode_indices'], list(range(1, 21)))
        self.assertEqual(report['excluded_episode_indices'], [0])
        self.assertFalse(report['test_used_for_fit'])
        self.assertAlmostEqual(report['comparisons']['learned_vs_original']['macro_gain'], -.15)

    def test_bootstrap_draws_paired_cases_independently_inside_each_style(self):
        data = rows()
        first = {style: [row['records']['learned'] for row in data if row['style'] == style]
                 for style in training.STYLES}
        second = {style: [row['records']['original'] for row in data if row['style'] == style]
                  for style in training.STYLES}
        choices = []
        rng = np.random.default_rng(31)

        def choose(values, shape, replace):
            choices.append((np.asarray(values).copy(), shape, replace))
            return rng.choice(values, shape, replace=replace)

        def style_statistics(a, b, samples):
            style = next(iter(a))
            self.assertEqual(set(a), {style})
            self.assertEqual(set(b), {style})
            return {'per_style': {style: {'only_first_saves': 1, 'only_second_saves': 2, 'p_exact': 1.}},
                    'macro_gain_ci95': [-1., 1.]}

        with patch.object(heldout, 'compare_styles', side_effect=style_statistics) as paired, \
                patch.object(heldout.np.random, 'default_rng', return_value=Mock(choice=choose)):
            result = heldout.paired_comparison(first, second, samples=123, seed=31)
        self.assertEqual(paired.call_count, 8)
        self.assertEqual(len(choices), 8)
        for differences, shape, replace in choices:
            self.assertEqual(shape, (123, 20))
            self.assertTrue(replace)
            self.assertEqual(differences.shape, (20,))
        self.assertAlmostEqual(result['macro_gain'], -.15)
        self.assertIn('independently within each style', result['ci_method'])

    def test_paired_statistics_reject_wrong_partner_seeds_and_duplicate_cases(self):
        first = {'center': [case('center', i)['records']['learned'] for i in range(1, 21)]}
        second = {'center': [case('center', i)['records']['original'] for i in range(1, 21)]}
        altered = copy.deepcopy(second)
        altered['center'][0]['seed'] += 999
        with self.assertRaises(ValueError):
            heldout.paired_comparison(first, altered, samples=20)
        with self.assertRaises(ValueError):
            heldout.paired_comparison(first, {'center': second['center']+[second['center'][0]]}, samples=20)


class HeldoutFrozenEvaluationTests(unittest.TestCase):
    def unfitted_fixture(self, directory):
        parent = Path(directory)
        output, model_path = parent/'run', parent/'selector.json'
        source_path = parent/'frozen_source_fixture.py'
        source_path.write_text('frozen_source = 1\n', encoding='utf-8')
        relative_source = source_path.relative_to(training.ROOT).as_posix()
        source = {'hashes': {relative_source: training._sha256(source_path)},
                  'versions': {'python': 'fixture'}, 'commit': 'fixture-parent',
                  'dirty': True, 'platform': 'fixture', 'bhuman_source_commit': 'original'}
        with patch.object(training, 'source_metadata', return_value=source):
            run = training.prepare_run(output, fit_epochs=3)
        return output, model_path, source_path, source, run

    def fixture(self, directory):
        output, model_path, source_path, source, run = self.unfitted_fixture(directory)
        fixed = protocol(run)
        training._atomic_json(output/'heldout_protocol.json', fixed)
        train, validation = [], []
        for split, records, offset in (('train', train, 100), ('validation', validation, 200)):
            for index in range(2):
                crossing = -.1+.2*index
                records.append({'style': 'center', 'seed': offset+index,
                    'context': {'version': CONTEXT_VERSION,
                                'values': [crossing, 1., 1., crossing, -1., 0., 0., 0., 0.]},
                    'eligible': True, 'targets': dict.fromkeys(DEFENSE_NAMES, index)})
        model = seal_model(fit_model(train, validation, regularizations=(.01,),
                                    margins=(.03,), epochs=3), {'fingerprint': run['fingerprint']})
        save_model(model, model_path)
        model_hash = training._sha256(model_path)
        training._atomic_json(output/'fit_report.json', {'fingerprint': run['fingerprint'],
            'model_sha256': model_hash, 'test_used_for_fit': False})
        evaluation_fingerprint = training._digest({'run': run['fingerprint'], 'model': model_hash,
                                                   'evaluation': training.EVALUATION_VERSION})
        training._atomic_json(output/'test_report.json', {'fingerprint': run['fingerprint'],
            'model_sha256': model_hash, 'evaluation_fingerprint': evaluation_fingerprint})
        return output, model_path, source_path, source, run, fixed, evaluation_fingerprint

    def evaluation_patches(self, source):
        return patch.object(training, 'source_metadata', return_value=source), \
            patch.object(heldout, '_jobs', side_effect=templates)

    def test_protocol_initializer_fixes_exact_scope_before_fit_without_loading_or_running_model(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            output, model_path, _, source, run = self.unfitted_fixture(directory)
            with patch.object(training, 'source_metadata', return_value=source), \
                    patch.object(heldout, 'load_model') as load, \
                    patch.object(heldout, '_execute_jobs') as execute:
                fixed = heldout.initialize_protocol(output)
                load.assert_not_called()
                execute.assert_not_called()
            self.assertEqual(heldout.validate_protocol(fixed, run), fixed)
            self.assertEqual(fixed['episode_indices'], list(range(1, 21)))
            self.assertEqual(fixed['excluded_episode_indices'], [0])
            self.assertFalse(fixed['model_weights_may_be_adjusted_with_test'])
            self.assertFalse(model_path.exists())
            self.assertFalse((output/'fit_report.json').exists())

    def test_protocol_initializer_preserves_existing_bytes_timestamp_and_reason(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            output, _, _, source, run = self.unfitted_fixture(directory)
            fixed = protocol(run)
            fixed['reason'] = 'A previously fixed declaration.'
            path = output/'heldout_protocol.json'
            training._atomic_json(path, fixed)
            original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns
            with patch.object(training, 'source_metadata', return_value=source):
                self.assertEqual(heldout.initialize_protocol(output), fixed)
                self.assertEqual(heldout.initialize_protocol(output), fixed)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(path.stat().st_mtime_ns, original_mtime)

    def test_protocol_initializer_refuses_post_fit_competing_run_or_changed_source(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            output, _, _, source, _, _, _ = self.fixture(directory)
            with patch.object(training, 'source_metadata', return_value=source):
                with self.assertRaisesRegex(ValueError, 'before fit_report'):
                    heldout.initialize_protocol(output)
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            output, _, _, source, run = self.unfitted_fixture(directory)
            fixed = protocol(run)
            fixed['run_fingerprint'] = 'another-run'
            path = output/'heldout_protocol.json'
            training._atomic_json(path, fixed)
            original_bytes = path.read_bytes()
            with patch.object(training, 'source_metadata', return_value=source):
                with self.assertRaisesRegex(ValueError, 'blind protocol differs'):
                    heldout.initialize_protocol(output)
            self.assertEqual(path.read_bytes(), original_bytes)
            changed_source = copy.deepcopy(source)
            changed_source['hashes'][next(iter(changed_source['hashes']))] = 'changed'
            with patch.object(training, 'source_metadata', return_value=changed_source):
                with self.assertRaisesRegex(ValueError, 'resume rejected'):
                    heldout.initialize_protocol(output)

    def test_reuses_152_cases_collects_only_eight_twenty_cases_and_preserves_exploratory_report(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            output, model_path, _, source, run, fixed, fingerprint = self.fixture(directory)
            for style in training.STYLES:
                for index in range(20):
                    row = case(style, index)
                    identity = {key: row[key] for key in ('split', 'style', 'seed', 'episode_index')}
                    training._checkpoint_write(training._checkpoint_path(output, 'test', style, row['seed']),
                                               fingerprint, identity, row)
            original_bytes = (output/'test_report.json').read_bytes()
            model_bytes = model_path.read_bytes()
            compute = Mock(side_effect=lambda job: case(job['style'], job['episode_index']))
            metadata_patch, jobs_patch = self.evaluation_patches(source)
            with metadata_patch, jobs_patch, patch.object(heldout, '_evaluate_job', compute), \
                    patch.object(training, 'fit_model', side_effect=AssertionError('blind test refitted')):
                report = heldout.evaluate_heldout(output, model_path, workers=1, progress=None)
            self.assertEqual(compute.call_count, 8)
            self.assertEqual({call.args[0]['episode_index'] for call in compute.call_args_list}, {20})
            self.assertEqual({call.args[0]['style'] for call in compute.call_args_list}, set(training.STYLES))
            self.assertEqual((output/'test_report.json').read_bytes(), original_bytes)
            self.assertEqual(model_path.read_bytes(), model_bytes)
            self.assertEqual(report['cached_cases_before_run'], 152)
            self.assertEqual(report['collected_cases_this_run'], 8)
            self.assertEqual(report['selected_cases'], 160)
            raw = json.loads((output/'selected_case_rows.json').read_text())
            self.assertEqual(raw['rows'], rows())
            self.assertEqual(report['selected_raw_rows']['sha256'],
                             training._sha256(output/'selected_case_rows.json'))
            self.assertEqual(report['summary']['learned']['macro_save_rate'], .3)

    def test_wrong_cached_model_fingerprint_is_rejected_before_duels(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            output, model_path, _, source, _, _, _ = self.fixture(directory)
            row = case('center', 1)
            identity = {key: row[key] for key in ('split', 'style', 'seed', 'episode_index')}
            path = training._checkpoint_path(output, 'test', 'center', row['seed'])
            training._checkpoint_write(path, 'wrong-model-fingerprint', identity, row)
            metadata_patch, jobs_patch = self.evaluation_patches(source)
            with metadata_patch, jobs_patch, patch.object(heldout, '_evaluate_job') as compute:
                with self.assertRaisesRegex(ValueError, 'identity/hash'):
                    heldout.evaluate_heldout(output, model_path, workers=1, progress=None)
                compute.assert_not_called()

    def test_model_or_protocol_modified_after_freezing_is_rejected_before_duels(self):
        for artifact in ('model', 'protocol_time'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
                output, model_path, _, source, _, fixed, _ = self.fixture(directory)
                if artifact == 'model':
                    model_path.write_text(model_path.read_text()+'\n')
                else:
                    fixed['created_utc'] = '2099-01-01T00:00:00+00:00'
                    training._atomic_json(output/'heldout_protocol.json', fixed)
                metadata_patch, jobs_patch = self.evaluation_patches(source)
                with metadata_patch, jobs_patch, patch.object(heldout, '_evaluate_job') as compute:
                    with self.assertRaises(ValueError):
                        heldout.evaluate_heldout(output, model_path, workers=1, progress=None)
                    compute.assert_not_called()

    def test_model_and_frozen_source_changes_during_test_are_rejected(self):
        for artifact in ('model', 'source'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
                output, model_path, source_path, source, _, _, _ = self.fixture(directory)

                def execute(*args, **kwargs):
                    path = model_path if artifact == 'model' else source_path
                    path.write_text(path.read_text()+'\n')
                    return rows()

                metadata_patch, jobs_patch = self.evaluation_patches(source)
                with metadata_patch, jobs_patch, patch.object(heldout, '_execute_jobs', side_effect=execute):
                    with self.assertRaisesRegex(RuntimeError, 'changed during the test'):
                        heldout.evaluate_heldout(output, model_path, workers=1, progress=None)


if __name__ == '__main__':
    unittest.main()
