"""Independent split, selection and artifact tests without native physics.

Motor duels are mocked; seed generation, bounded candidates, paired-prefix
checks, cache serialization, aggregation, selection and freezing execute.
"""
import copy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from penalty.keeper_selector import CONTEXT_VERSION
from penalty.rl import motion_training as training
from penalty.skills.keeper_motion import MotionParameters, ParameterizedKeeperMotion


def source_metadata():
    return {'hashes': {}, 'versions': {'python': 'test', 'numpy': 'test'},
            'commit': 'test-parent', 'bhuman_source_commit': 'original'}


def record(seed, result='save', committed=True):
    return {'seed': seed, 'result': result, 'early_dive': False,
            'selector_context': {'version': CONTEXT_VERSION,
                                 'values': [.1, 1.12, 1., .1, -1., 0., .01, 0., 0.]}
                                 if committed else None,
            'first_trigger': {'time_s': .2, 'capture_s': .2, 'core_side': 0,
                              'crossing_y': .1, 'time_to_line': 1.12}
                             if committed else None,
            'commit_time': .2 if committed else None,
            'commit_side': 0, 'commit_defense': 'center' if committed else None,
            'requested_defense': 'center' if committed else None,
            'no_commit': not committed,
            'reset_info': {'engine_seed': seed, 'ball_xy': [1., 0.],
                           'start_offsets': {'keeper': [.01, 0.], 'striker': [0., .01]}}}


def case(job, outcomes=None, committed=True):
    records = {profile['id']: record(job['seed'],
                                    (outcomes or {}).get(profile['id'], 'save'), committed)
               for profile in job['profiles']}
    return {key: job[key] for key in ('split', 'style', 'seed', 'episode_index')} | {
        'records': records, 'pairing': training.motion_pairing(records),
        'profile_order': [profile['id'] for profile in job['profiles']], 'wall_s': 0.}


class MotionProtocolTests(unittest.TestCase):
    def test_all_style_indices_are_unique_and_splits_are_disjoint(self):
        seeds = {split: {training.episode_seed(split, style, index)
                         for style in training.STYLES for index in (0, 1, 9999)}
                 for split in training.SPLIT_BASES}
        for split, values in seeds.items():
            self.assertEqual(len(values), len(training.STYLES)*3)
            for other, other_values in seeds.items():
                if split != other:
                    self.assertFalse(values & other_values)
        for split, style, index in (('smoke', 'center', 0), ('train', 'unknown', 0),
                                     ('train', 'center', -1), ('test', 'center', 10000),
                                     ('test', 'center', True), ('test', 'center', .5)):
            with self.subTest(split=split, style=style, index=index):
                with self.assertRaises(ValueError):
                    training.episode_seed(split, style, index)

    def test_candidates_are_bounded_reproducible_and_include_exact_nominal(self):
        profiles = training.candidates()
        self.assertEqual(profiles, training.candidates())
        self.assertEqual(len(profiles), 16)
        self.assertEqual(len({profile['id'] for profile in profiles}), 16)
        self.assertEqual(profiles[0], {'id': 'nominal', 'parameters': asdict(MotionParameters())})
        for profile in profiles:
            ParameterizedKeeperMotion.validate(MotionParameters(**profile['parameters']))
        changed = training.candidates(seed=99001)
        self.assertEqual(profiles[:5], changed[:5])
        self.assertNotEqual(profiles[5:], changed[5:])
        for count in (0, 4, 65, True, 5.5):
            with self.assertRaises(ValueError):
                training.candidates(count)

    def test_manifest_resume_binds_configuration_sources_and_runtime(self):
        settings = dict(count=5, train_episodes=1, validation_episodes=1, test_episodes=1)
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            with patch.object(training, '_sources', return_value=source_metadata()):
                manifest = training.prepare_run(directory, **settings)
                self.assertTrue(manifest['test_protocol_fixed_before_training'])
                self.assertFalse(manifest['configuration']['test_may_adjust_parameters'])
                self.assertEqual(training.prepare_run(directory, resume=True, **settings), manifest)
                with self.assertRaises(ValueError):
                    training.prepare_run(directory, **settings)
                with self.assertRaises(ValueError):
                    training.prepare_run(directory, resume=True,
                                         **dict(settings, test_episodes=2))
            altered = source_metadata()
            altered['versions']['numpy'] = 'different'
            with patch.object(training, '_sources', return_value=altered):
                with self.assertRaises(ValueError):
                    training.prepare_run(directory, resume=True, **settings)

    def test_resume_rejects_manifest_payload_tamper_with_unchanged_fingerprint(self):
        settings = dict(count=5, train_episodes=1, validation_episodes=1, test_episodes=1)
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory, \
                patch.object(training, '_sources', return_value=source_metadata()):
            manifest = training.prepare_run(directory, **settings)
            manifest['configuration']['episode_indices']['test'].append(20)
            training._atomic_json(Path(directory)/'run_manifest.json', manifest)
            with self.assertRaises(ValueError):
                training.prepare_run(directory, resume=True, **settings)

    def test_source_changes_abort_even_when_configuration_is_otherwise_identical(self):
        manifest = {'configuration': {'source_hashes': {'native.exe': 'fixed'}}}
        with patch.object(training, '_sha256', return_value='fixed'):
            training.check_sources(manifest)
        with patch.object(training, '_sha256', return_value='changed'):
            with self.assertRaises(RuntimeError):
                training.check_sources(manifest)


class MotionPairingTests(unittest.TestCase):
    def test_different_physical_results_can_share_identical_pre_motion_prefix(self):
        records = {'a': record(17, 'save'), 'b': record(17, 'goal')}
        self.assertTrue(training.motion_pairing(records)['paired'])
        self.assertFalse(training.motion_pairing(records)['no_commit'])

    def test_prefix_reset_seed_and_motor_commit_disagreements_are_rejected(self):
        changes = (
            lambda row: row['selector_context']['values'].__setitem__(0, .2),
            lambda row: row['first_trigger'].__setitem__('time_s', .212),
            lambda row: row.__setitem__('requested_defense', 'left'),
            lambda row: row.__setitem__('commit_defense', 'left'),
            lambda row: row['reset_info'].__setitem__('engine_seed', 18),
            lambda row: row['reset_info'].__setitem__('ball_xy', [1., .01]),
            lambda row: row['reset_info'].__setitem__('start_offsets', {}),
            lambda row: row.__setitem__('commit_time', .212),
            lambda row: row.__setitem__('seed', 18),
            lambda row: row.__setitem__('no_commit', True),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                records = {'a': record(17), 'b': record(17)}
                change(records['b'])
                self.assertFalse(training.motion_pairing(records)['paired'])

    def test_empty_sim_error_and_partial_commit_cannot_form_a_valid_pair(self):
        for records in ({}, {'a': record(17), 'b': record(17, 'sim_error')},
                        {'a': record(17), 'b': record(17, 'timeout', False)}):
            self.assertFalse(training.motion_pairing(records)['paired'])

    def test_no_commit_cases_stay_in_rate_and_count_denominators(self):
        profiles = training.candidates(5)[:2]
        rows = []
        for style in training.STYLES:
            for index, committed in enumerate((True, False)):
                job = {'split': 'train', 'style': style, 'episode_index': index,
                       'seed': training.episode_seed('train', style, index), 'profiles': profiles}
                rows.append(case(job, None if committed else
                                 dict.fromkeys((p['id'] for p in profiles), 'timeout'), committed))
        summary, groups = training.summarize(rows, profiles)
        self.assertEqual(summary['paired_cases'], 16)
        self.assertEqual(summary['duels'], 32)
        for profile in profiles:
            name = profile['id']
            self.assertEqual(summary['summary'][name]['macro_save_rate'], .5)
            self.assertEqual(summary['save_counts'][name], 8)
            self.assertEqual(summary['no_commit_counts'][name], 8)
            self.assertEqual(summary['defense_counts'][name]['none'], 8)
            for style in training.STYLES:
                self.assertEqual(len(groups[name][style]), 2)
                self.assertEqual(summary['summary'][name]['per_style'][style]['episodes'], 2)

    def test_incomplete_or_unpaired_group_cannot_be_summarized(self):
        profiles = training.candidates(5)[:2]
        job = {'split': 'train', 'style': 'center', 'episode_index': 0,
               'seed': 17, 'profiles': profiles}
        row = case(job)
        row['records'].pop(profiles[1]['id'])
        with self.assertRaises(ValueError):
            training.summarize([row], profiles)
        row = case(job)
        row['pairing']['paired'] = False
        with self.assertRaises(ValueError):
            training.summarize([row], profiles)


class MotionCheckpointTests(unittest.TestCase):
    def test_checkpoint_binds_run_case_identity_and_payload(self):
        identity = {'split': 'train', 'style': 'center', 'seed': 17, 'episode_index': 0}
        payload = dict(identity, records={'nominal': record(17)})
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            training._checkpoint(directory, identity, 'run-a', payload)
            self.assertEqual(training._checkpoint(directory, identity, 'run-a'), payload)
            with self.assertRaises(ValueError):
                training._checkpoint(directory, identity, 'run-b')
            with self.assertRaises(ValueError):
                training._checkpoint(directory, identity, 'run-a', dict(payload, seed=18))
            path = Path(directory)/'checkpoints/train/center/17.json'
            envelope = training._read_json(path)
            envelope['payload']['records']['nominal']['result'] = 'goal'
            training._atomic_json(path, envelope)
            with self.assertRaises(ValueError):
                training._checkpoint(directory, identity, 'run-a')

    def test_rehashed_payload_still_must_match_the_envelope_identity(self):
        identity = {'split': 'train', 'style': 'center', 'seed': 17, 'episode_index': 0}
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            training._checkpoint(directory, identity, 'run-a', dict(identity, records={}))
            path = Path(directory)/'checkpoints/train/center/17.json'
            envelope = training._read_json(path)
            envelope['payload']['seed'] = 18
            envelope['payload_sha256'] = training._digest(envelope['payload'])
            training._atomic_json(path, envelope)
            with self.assertRaises(ValueError):
                training._checkpoint(directory, identity, 'run-a')

    def test_resume_reads_complete_cases_without_rerunning_duels(self):
        jobs = [{'split': 'train', 'style': 'center', 'seed': 17+i, 'episode_index': i}
                for i in range(2)]
        compute = Mock(side_effect=lambda job: dict(job, records={}))
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            first = training._execute(jobs, compute, directory, 'run-a', 1, progress=None)
            self.assertEqual(compute.call_count, 2)
            compute.reset_mock()
            self.assertEqual(training._execute(jobs, compute, directory, 'run-a', 2,
                                              progress=None), first)
            compute.assert_not_called()


class MotionSelectionTests(unittest.TestCase):
    def new_manifest(self, directory):
        with patch.object(training, '_sources', return_value=source_metadata()):
            return training.prepare_run(directory, count=5, train_episodes=2,
                                        validation_episodes=2, test_episodes=2)

    def prepare_training(self, directory, manifest):
        profiles = manifest['configuration']['candidates']
        training._atomic_json(Path(directory)/'safety_report.json', {
            'fingerprint': manifest['fingerprint'],
            'eligible_profiles': [p['id'] for p in profiles]})
        with patch.object(training, 'player_spec', return_value=(None, {})), \
                patch.object(training, '_duel_job', side_effect=self.duel):
            return training.search(directory, manifest, workers=1, progress=None)

    @staticmethod
    def duel(job):
        if job['split'] == 'train':
            outcomes = {'candidate_01': 'save', 'candidate_02': 'save'
                        if job['episode_index'] == 0 else 'goal'}
        elif job['split'] == 'validation':
            outcomes = {'candidate_02': 'save', 'nominal': 'save'
                        if job['episode_index'] == 0 else 'goal'}
        else:
            outcomes = {'trained': 'goal', 'balanced': 'save'}
        return case(job, {p['id']: outcomes.get(p['id'], 'goal') for p in job['profiles']})

    def fit(self, directory, manifest, model_path):
        with patch.object(training, 'player_spec', return_value=(None, {})), \
                patch.object(training, '_duel_job', side_effect=self.duel):
            return training.validate_and_fit(directory, manifest, model_path,
                                             workers=1, progress=None)

    def test_ranking_prioritizes_macro_then_worst_style_then_nominal(self):
        profiles = training.candidates(5)
        report = {'summary': {p['id']: {'macro_save_rate': .5,
                            'worst_style_save_rate': .25, 'sim_errors': 0} for p in profiles}}
        self.assertEqual(training.rank_profiles(profiles, report)[0]['id'], 'nominal')
        report['summary']['candidate_01']['worst_style_save_rate'] = .4
        self.assertEqual(training.rank_profiles(profiles, report)[0]['id'], 'candidate_01')
        report['summary']['candidate_02']['macro_save_rate'] = .6
        self.assertEqual(training.rank_profiles(profiles, report)[0]['id'], 'candidate_02')
        report['summary']['nominal']['sim_errors'] = 1
        with self.assertRaises(ValueError):
            training.rank_profiles(profiles, report)

    def test_safety_job_unpacks_real_validator_api_and_hashes_every_trace(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            job = {'split': 'safety', 'style': 'all_defenses', 'seed': 80002000,
                   'episode_index': 0, 'profile': training.candidates(5)[0],
                   'output': directory}
            def validated(parameters, defense, seed):
                self.assertEqual(parameters, MotionParameters())
                self.assertEqual(seed, job['seed'])
                return ({'defense': defense, 'passed': defense != 'right',
                         'children_collected': True},
                        [{'time_s': .012, 'defense': defense}])
            with patch('tools.validate_keeper_motion.validate_motion_case',
                       side_effect=validated) as validate:
                row = training._safety_job(job)
            self.assertEqual(validate.call_count, 6)
            self.assertEqual({c['defense'] for c in row['cases']}, set(training.DEFENSE_NAMES))
            self.assertFalse(row['passed'])
            for case_summary in row['cases']:
                path = Path(directory)/case_summary['trace']
                self.assertEqual(case_summary['trace_sha256'], training._sha256(path))
                self.assertEqual(case_summary['trace_frames'], 1)
                self.assertEqual(json.loads(gzip.decompress(path.read_bytes())),
                                 {'time_s': .012, 'defense': case_summary['defense']})

    def test_failed_nominal_recovery_gate_aborts_before_search(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            profiles = manifest['configuration']['candidates']
            rows = [{'profile': p, 'passed': p['id'] != 'nominal'} for p in profiles]
            with patch.object(training, '_execute', return_value=rows):
                with self.assertRaisesRegex(RuntimeError, 'nominal'):
                    training.safety_screen(directory, manifest, workers=1, progress=None)
            self.assertFalse((Path(directory)/'safety_report.json').exists())

    def test_only_safe_train_shortlist_and_validation_choose_frozen_profile(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            train = self.prepare_training(directory, manifest)
            self.assertEqual([p['id'] for p in train['shortlist']],
                             ['candidate_01', 'candidate_02', 'nominal'])
            # An unrelated test artifact cannot choose the fitted candidate.
            training._atomic_json(Path(directory)/'test_report.json', {
                'selected_profile': manifest['configuration']['candidates'][4]})
            model_path = Path(directory)/'profile.json'
            fit = self.fit(directory, manifest, model_path)
            model = training.load_profile(model_path)
            self.assertEqual(fit['selected_profile']['id'], 'candidate_02')
            self.assertEqual(model['selected_candidate'], 'candidate_02')
            self.assertEqual(model['training_fingerprint'], manifest['fingerprint'])
            self.assertEqual(model['train_report_sha256'],
                             training._sha256(Path(directory)/'train_report.json'))
            self.assertFalse(fit['test_used_for_fit'])
            self.assertFalse(model['test_used_for_fit'])
            self.assertTrue(model['simulation_only'])

    def test_unsafe_profile_is_not_sent_to_training_duels(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            profiles = manifest['configuration']['candidates']
            safe = [p['id'] for p in profiles if p['id'] != 'candidate_01']
            training._atomic_json(Path(directory)/'safety_report.json', {
                'fingerprint': manifest['fingerprint'], 'eligible_profiles': safe})
            seen = []
            def duel(job):
                seen.extend(p['id'] for p in job['profiles'])
                return self.duel(job)
            with patch.object(training, 'player_spec', return_value=(None, {})), \
                    patch.object(training, '_duel_job', side_effect=duel):
                train = training.search(directory, manifest, workers=1, progress=None)
            self.assertNotIn('candidate_01', seen)
            self.assertNotIn('candidate_01', [p['id'] for p in train['shortlist']])
            self.assertIn('nominal', [p['id'] for p in train['shortlist']])

    def test_model_integrity_scope_and_runtime_bounds_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            self.prepare_training(directory, manifest)
            model_path = Path(directory)/'profile.json'
            self.fit(directory, manifest, model_path)
            original = training._read_json(model_path)
            changed = copy.deepcopy(original)
            changed['parameters']['arm_spread_rad'] = .01
            training._atomic_json(model_path, changed)
            with self.assertRaises(ValueError):
                training.load_profile(model_path)
            for field, value in (('test_used_for_fit', True), ('simulation_only', False)):
                changed = copy.deepcopy(original)
                changed[field] = value
                changed.pop('integrity_sha256')
                changed['integrity_sha256'] = training._digest(changed)
                training._atomic_json(model_path, changed)
                with self.assertRaises(ValueError):
                    training.load_profile(model_path)
            changed = copy.deepcopy(original)
            changed['parameters']['jump_time_scale'] = 10.
            changed.pop('integrity_sha256')
            changed['integrity_sha256'] = training._digest(changed)
            training._atomic_json(model_path, changed)
            with self.assertRaises(ValueError):
                training.load_profile(model_path)

    def test_blind_test_is_fixed_paired_and_cannot_retune_a_losing_profile(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            self.prepare_training(directory, manifest)
            model_path = Path(directory)/'profile.json'
            self.fit(directory, manifest, model_path)
            before = model_path.read_bytes()
            seen = []
            def duel(job):
                seen.append(job)
                return self.duel(job)
            with patch.object(training, 'player_spec', return_value=(None, {})), \
                    patch.object(training, '_duel_job', side_effect=duel), \
                    patch.object(training, 'rank_profiles', side_effect=AssertionError('test ranked profiles')):
                report = training.blind_test(directory, manifest, model_path,
                                              workers=1, progress=None)
            self.assertEqual(len(seen), 16)
            self.assertEqual({j['split'] for j in seen}, {'test'})
            self.assertEqual(report['duels'], 32)
            self.assertEqual(report['save_counts'], {'trained': 0, 'balanced': 16})
            self.assertEqual(report['comparisons']['trained_vs_balanced']['macro_gain'], -1.)
            self.assertEqual(before, model_path.read_bytes())
            self.assertTrue(report['model_frozen_before_test'])
            self.assertFalse(report['test_used_for_fit'])
            self.assertFalse(report['hardware_validated'])

    def test_blind_test_refuses_changed_frozen_model_or_validation_artifact(self):
        for artifact in ('profile.json', 'validation_report.json'):
            with self.subTest(artifact=artifact), \
                    tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
                manifest = self.new_manifest(directory)
                self.prepare_training(directory, manifest)
                model_path = Path(directory)/'profile.json'
                self.fit(directory, manifest, model_path)
                path = Path(directory)/artifact
                path.write_bytes(path.read_bytes()+b'\n')
                with patch.object(training, '_execute') as execute:
                    with self.assertRaises(ValueError):
                        training.blind_test(directory, manifest, model_path,
                                            workers=1, progress=None)
                    execute.assert_not_called()

    def test_blind_test_refuses_model_mutation_during_evaluation(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as directory:
            manifest = self.new_manifest(directory)
            self.prepare_training(directory, manifest)
            model_path = Path(directory)/'profile.json'
            self.fit(directory, manifest, model_path)
            def duel(job):
                model_path.write_bytes(model_path.read_bytes()+b'\n')
                return self.duel(job)
            with patch.object(training, 'player_spec', return_value=(None, {})), \
                    patch.object(training, '_duel_job', side_effect=duel):
                with self.assertRaises(RuntimeError):
                    training.blind_test(directory, manifest, model_path,
                                        workers=1, progress=None)


if __name__ == '__main__':
    unittest.main()
