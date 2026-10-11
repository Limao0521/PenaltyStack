"""Independent tests of physical-label grouping and the frozen test split.

Expensive native duels are mocked; the real selector, fitting, checkpoint
serialization and final aggregation still execute where those matter.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from penalty.keeper_defenses import DEFENSE_NAMES
from penalty.keeper_selector import CONTEXT_VERSION, ForcedDefenseKeeper, load_model
from penalty.rl.contracts import OBSERVATION_FIELDS, StrikerAction
from penalty.rl import selector_training as training


def context(crossing=.1):
    return {'version': CONTEXT_VERSION,
            'values': [crossing, 1.12, 1., crossing, -1., 0., .01, 0., 0.]}


def observation(time):
    return {'version': 'vision_state_v2', 'time_s': time, 'capture_s': time,
            'remaining_time': 28.-time,
            'ball': {'visible': True, 'position_valid': True, 'velocity_valid': True,
                     'position': [1., .1], 'velocity': [-1., 0.], 'age_s': .01,
                     'velocity_covariance': [[0., 0.], [0., 0.]]}}


def record(defense, seed=17, result='save', committed=True):
    return {'seed': seed, 'result': result, 'selector_context': context() if committed else None,
            'first_trigger': {'time_s': .2, 'capture_s': .2, 'core_side': 0,
                              'crossing_y': .1, 'time_to_line': 1.12} if committed else None,
            'commit_time': .2 if committed else None,
            'commit_defense': defense if committed else None,
            'requested_defense': defense if committed else None,
            'context_digest': training._digest(context()) if committed else None,
            'no_commit': not committed, 'early_dive': False, 'commit_side': 0}


def case(split, style, index, crossing=.1):
    seed = training.episode_seed(split, style, index)
    outcomes = {name: record(name, seed, 'save' if offset % 2 else 'goal')
                for offset, name in enumerate(DEFENSE_NAMES)}
    for row in outcomes.values():
        row['selector_context'] = context(crossing)
        row['first_trigger']['crossing_y'] = crossing
    return {'dataset': training.DATASET_VERSION, 'split': split, 'style': style,
            'seed': seed, 'episode_index': index, 'context': context(crossing),
            'eligible': True, 'targets': {name: int(row['result'] == 'save')
                                         for name, row in outcomes.items()},
            'outcomes': outcomes, 'pairing': training.pairing_check(outcomes)}


def source_metadata():
    return {'hashes': {'native.exe': 'native-v1', 'striker/model.npz': 'striker-v1'},
            'versions': {'python': 'test', 'numpy': 'test'}, 'commit': 'test-parent',
            'dirty': True, 'platform': 'test', 'bhuman_source_commit': 'original'}


class SelectorCollectionTests(unittest.TestCase):
    def collect_mocked(self, transform=None):
        seen = []

        def episode(world, camera, striker, keeper, seed):
            seen.append((seed, keeper.defense, id(world), id(camera)))
            row = record(keeper.defense, seed)
            return transform(keeper.defense, row) if transform else row

        world, camera = Mock(), Mock()
        job = {'split': 'train', 'style': 'center', 'seed': 17,
               'episode_index': 0, 'trained': {}}
        with patch.object(training, 'SensorDefenseDuelWorld', return_value=world), \
                patch.object(training, '_camera', return_value=camera), \
                patch.object(training, '_striker', return_value=Mock()), \
                patch.object(training, 'run_camera_episode', side_effect=episode):
            row = training._collect_job(job)
        self.assertEqual(set(row['outcomes']), set(DEFENSE_NAMES))
        self.assertTrue(world.close.called)
        self.assertTrue(camera.close.called)
        return row, seen

    def test_six_motor_counterfactuals_form_one_camera_context(self):
        row, seen = self.collect_mocked()
        self.assertEqual(len(seen), 6)
        self.assertEqual({seed for seed, _, _, _ in seen}, {17})
        self.assertEqual({defense for _, defense, _, _ in seen}, set(DEFENSE_NAMES))
        self.assertEqual(len({world for _, _, world, _ in seen}), 1)
        self.assertTrue(row['eligible'])
        self.assertEqual(row['context'], context())
        self.assertEqual(row['targets'], dict.fromkeys(DEFENSE_NAMES, 1))
        self.assertEqual({r['commit_time'] for r in row['outcomes'].values()}, {.2})

    def test_context_and_commit_time_mismatches_reject_all_six_labels(self):
        def changed_context(defense, row):
            if defense == 'wide_left':
                row['selector_context']['values'][0] += .001
            return row

        def changed_commit(defense, row):
            if defense == 'wide_left':
                row['commit_time'] += .012
            return row

        for transform in (changed_context, changed_commit):
            with self.subTest(transform=transform.__name__):
                row, _ = self.collect_mocked(transform)
                self.assertFalse(row['eligible'])
                self.assertIsNone(row['context'])
                self.assertEqual(row['targets'], dict.fromkeys(DEFENSE_NAMES, None))
                self.assertEqual(len(row['outcomes']), 6)

    def test_one_sim_error_cannot_become_six_valid_training_examples(self):
        def failed(defense, row):
            return training._error_record(row['seed'], RuntimeError('engine stopped')) \
                if defense == 'wide_right' else row
        row, _ = self.collect_mocked(failed)
        self.assertFalse(row['eligible'])
        self.assertEqual(row['targets'], dict.fromkeys(DEFENSE_NAMES, None))
        summary = training.dataset_summary([row])
        self.assertEqual(summary['duels'], 6)
        self.assertEqual(summary['eligible_contexts'], 0)
        self.assertEqual(summary['sim_errors'], 1)
        self.assertEqual(summary['training_labels_by_defense'], dict.fromkeys(DEFENSE_NAMES, 0))

    def test_no_commit_case_remains_in_dataset_without_fabricated_context(self):
        def no_commit(defense, row):
            return record(defense, row['seed'], 'timeout', committed=False)
        row, _ = self.collect_mocked(no_commit)
        self.assertFalse(row['eligible'])
        self.assertEqual(row['pairing']['reason'], 'no_commit')
        self.assertEqual(row['targets'], dict.fromkeys(DEFENSE_NAMES, None))
        summary = training.dataset_summary([row])
        self.assertEqual(summary['no_commit_duels'], 6)
        self.assertEqual(summary['no_commit_contexts'], 1)
        self.assertEqual(summary['duels'], 6)

    def test_partial_commit_and_core_side_disagreement_are_rejected(self):
        for field in ('missing_commit', 'core_side'):
            rows = {name: record(name) for name in DEFENSE_NAMES}
            if field == 'missing_commit':
                rows['right'] = record('right', committed=False)
            else:
                rows['right']['first_trigger']['core_side'] = -1
            self.assertFalse(training.pairing_check(rows)['paired'])
        with self.assertRaises(ValueError):
            training.pairing_check({'center': record('center')})

    def test_group_rejects_wrong_executed_defense_and_nonfinite_context(self):
        rows = {name: record(name) for name in DEFENSE_NAMES}
        rows['right']['commit_defense'] = 'left'
        self.assertFalse(training.pairing_check(rows)['paired'])
        rows = {name: record(name) for name in DEFENSE_NAMES}
        rows['right']['selector_context']['values'][0] = float('nan')
        self.assertFalse(training.pairing_check(rows)['paired'])

    def test_first_context_is_copied_before_the_defense_executes(self):
        states = [observation(.1), observation(.2)]
        camera = Mock()
        camera.observe.side_effect = states
        striker = Mock(return_value=StrikerAction(0., 0, .5, 1.).to_normalized())
        vector = SimpleNamespace(to_array=lambda: np.zeros(len(OBSERVATION_FIELDS)))
        world = Mock()
        world.sim = SimpleNamespace(bridges={'keeper': SimpleNamespace(hello={
            'capabilities': [training.DEFENSE_CATALOG_VERSION],
            'keeper_defenses': list(DEFENSE_NAMES)})})
        world.reset.return_value = (vector, None, {'reset': 'same physical seed'})
        executed = []

        def step(striker_action, keeper_action):
            executed.append(keeper_action)
            if keeper_action.fires:
                # A later sensor mutation cannot alter the retained context.
                states[1]['ball']['position'][1] = 999.
                return vector, None, 'save', {}
            return vector, None, None, {}

        world.duel_step.side_effect = step
        world.record.return_value = record('wide_right', 17)
        frames = []
        with patch.object(training, 'PerceptionNoise', return_value=lambda obs, rng: obs):
            row = training.run_camera_episode(world, camera, striker,
                ForcedDefenseKeeper('wide_right'), 17,
                on_frame=lambda state, action, decision: frames.append((action.fires, len(executed))))
        self.assertEqual(frames, [(False, 0), (True, 1)])
        self.assertEqual(row['selector_context'], context())
        self.assertEqual(row['first_trigger']['time_s'], .2)
        self.assertEqual(row['requested_defense'], 'wide_right')
        self.assertEqual(row['steps'], 2)
        self.assertFalse(row['no_commit'])
        self.assertTrue(camera.reset.called)
        self.assertTrue(striker.reset.called)


class SelectorCheckpointTests(unittest.TestCase):
    def test_seed_sets_are_disjoint_for_all_splits_styles_and_indices(self):
        seeds = {split: {training.episode_seed(split, style, index)
                         for style in training.STYLES for index in (0, 1, 9999)}
                 for split in training.SPLIT_BASES}
        for first in seeds:
            self.assertEqual(len(seeds[first]), len(training.STYLES)*3)
            for second in seeds:
                if first != second:
                    self.assertFalse(seeds[first] & seeds[second])
        for split, style, index in (('unknown', 'center', 0), ('train', 'unknown', 0),
                                     ('train', 'center', -1), ('train', 'center', 10000)):
            with self.assertRaises(ValueError):
                training.episode_seed(split, style, index)

    def test_resume_requires_exact_source_models_runtime_and_configuration(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            settings = dict(train_episodes=1, validation_episodes=1, test_episodes=1,
                            styles=('center',), fit_epochs=2)
            source = source_metadata()
            with patch.object(training, 'source_metadata', return_value=source):
                original = training.prepare_run(directory, **settings)
                self.assertEqual(training.prepare_run(directory, resume=True, **settings), original)
                with self.assertRaisesRegex(ValueError, 'already contains'):
                    training.prepare_run(directory, **settings)
                with self.assertRaisesRegex(ValueError, 'resume rejected'):
                    training.prepare_run(directory, resume=True, **dict(settings, fit_epochs=3))
            for section, key in (('hashes', 'native.exe'), ('hashes', 'striker/model.npz'),
                                 ('versions', 'numpy')):
                altered = copy.deepcopy(source)
                altered[section][key] = 'changed'
                with patch.object(training, 'source_metadata', return_value=altered):
                    with self.assertRaisesRegex(ValueError, 'resume rejected'):
                        training.prepare_run(directory, resume=True, **settings)

    def test_checkpoints_bind_payload_hash_run_and_case_identity(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            path = Path(directory)/'case.json'
            payload = case('train', 'center', 0)
            identity = {key: payload[key] for key in ('split', 'style', 'seed', 'episode_index')}
            training._checkpoint_write(path, 'run-a', identity, payload)
            self.assertEqual(training._checkpoint_read(path, 'run-a', identity), payload)
            with self.assertRaisesRegex(ValueError, 'identity/hash'):
                training._checkpoint_read(path, 'run-b', identity)
            with self.assertRaisesRegex(ValueError, 'identity/hash'):
                training._checkpoint_read(path, 'run-a', dict(identity, seed=identity['seed']+1))
            envelope = json.loads(path.read_text())
            envelope['payload']['targets']['center'] = 99
            path.write_text(json.dumps(envelope))
            with self.assertRaisesRegex(ValueError, 'identity/hash'):
                training._checkpoint_read(path, 'run-a', identity)

    def test_valid_hash_cannot_hide_payload_from_another_case(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            path = Path(directory)/'case.json'
            payload = case('train', 'center', 0)
            identity = {key: payload[key] for key in ('split', 'style', 'seed', 'episode_index')}
            payload['seed'] += 1
            with self.assertRaises(ValueError):
                training._checkpoint_write(path, 'run-a', identity, payload)

    def test_resume_reuses_complete_cases_without_rerunning_duels(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            rows = [case('train', 'center', index) for index in range(2)]
            jobs = [{key: row[key] for key in ('split', 'style', 'seed', 'episode_index')}
                    for row in rows]
            compute = Mock(side_effect=lambda job: rows[job['episode_index']])
            first = training._execute_jobs(jobs, compute, directory, 'run-a', workers=1, progress=None)
            self.assertEqual(compute.call_count, 2)
            compute.reset_mock()
            second = training._execute_jobs(jobs, compute, directory, 'run-a', workers=2, progress=None)
            self.assertEqual(first, second)
            compute.assert_not_called()


class SelectorFitAndEvaluationTests(unittest.TestCase):
    def new_manifest(self, directory, episodes=2):
        with patch.object(training, 'source_metadata', return_value=source_metadata()):
            return training.prepare_run(directory, train_episodes=episodes,
                validation_episodes=episodes, test_episodes=episodes,
                styles=('center', 'mixed'), fit_epochs=3)

    def collect_datasets(self, directory, manifest):
        def job(spec):
            return case(spec['split'], spec['style'], spec['episode_index'],
                        crossing=-.2+.2*spec['episode_index'])
        with patch.object(training, '_collect_job', side_effect=job):
            return training.collect(directory, manifest, workers=1, progress=None)

    def test_fit_roundtrip_seals_dataset_and_source_provenance_before_test(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            self.collect_datasets(directory, manifest)
            path = Path(directory)/'selector.json'
            frozen = training.fit(directory, manifest, path, progress=None)
            model = load_model(path)
            self.assertFalse(frozen['test_used_for_fit'])
            self.assertFalse(model['training_provenance']['test_used_for_fit'])
            self.assertEqual(model['training_provenance']['fingerprint'], manifest['fingerprint'])
            self.assertEqual(frozen['model_sha256'], training._sha256(path))
            self.assertEqual(model['fit_report']['training_rows'], 4)
            self.assertEqual(model['fit_report']['validation_rows'], 4)

    def test_dataset_edit_cannot_bypass_independently_hashed_checkpoints(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            self.collect_datasets(directory, manifest)
            path = Path(directory)/'train_dataset.json'
            dataset = json.loads(path.read_text())
            dataset['rows'][0]['targets']['center'] = 1
            path.write_text(json.dumps(dataset))
            with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
                training.fit(directory, manifest, Path(directory)/'selector.json', progress=None)

    def test_final_pairs_three_policies_and_keeps_errors_and_no_commit_in_denominator(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            self.collect_datasets(directory, manifest)
            model_path = Path(directory)/'selector.json'
            training.fit(directory, manifest, model_path, progress=None)
            seen = []

            def evaluated(job):
                seen.append(job)
                index = ('center', 'mixed').index(job['style'])*2+job['episode_index']
                results = {'learned': ('save', 'sim_error', 'goal', 'timeout'),
                           'original': ('save', 'goal', 'save', 'timeout'),
                           'balanced': ('goal', 'save', 'save', 'timeout')}
                records = {mode: record('center', job['seed'], values[index],
                                         committed=values[index] not in ('sim_error', 'timeout'))
                           for mode, values in results.items()}
                return {'evaluation': training.EVALUATION_VERSION, 'split': 'test',
                        'style': job['style'], 'seed': job['seed'],
                        'episode_index': job['episode_index'], 'records': records}

            with patch.object(training, '_evaluate_job', side_effect=evaluated), \
                    patch.object(training, 'fit_model', side_effect=AssertionError('test refitted model')):
                report = training.evaluate(directory, manifest, model_path, workers=1, progress=None)
            self.assertEqual(len(seen), 4)
            self.assertEqual(set(report['records']), {'learned', 'original', 'balanced'})
            self.assertEqual(report['summary']['learned']['macro_save_rate'], .25)
            self.assertEqual(report['summary']['original']['macro_save_rate'], .5)
            self.assertEqual(report['summary']['balanced']['macro_save_rate'], .5)
            self.assertEqual(report['summary']['learned']['sim_errors'], 1)
            self.assertEqual(report['no_commit_counts']['learned'], 2)
            self.assertEqual(report['defense_counts']['learned']['none'], 2)
            for style in ('center', 'mixed'):
                seeds = [[row['seed'] for row in report['records'][mode][style]]
                         for mode in training.EVALUATION_MODES]
                self.assertEqual(seeds[0], seeds[1])
                self.assertEqual(seeds[1], seeds[2])
                self.assertEqual(report['summary']['learned']['per_style'][style]['episodes'], 2)
            self.assertEqual(report['comparisons']['learned_vs_original']['macro_gain'], -.25)
            self.assertTrue(report['model_frozen_before_test'])
            self.assertFalse(report['test_used_for_fit'])

    def test_final_refuses_weights_changed_after_fit(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            self.collect_datasets(directory, manifest)
            model_path = Path(directory)/'selector.json'
            training.fit(directory, manifest, model_path, progress=None)
            model_path.write_text(model_path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'exact model frozen'):
                training.evaluate(directory, manifest, model_path, workers=1, progress=None)


if __name__ == '__main__':
    unittest.main()
