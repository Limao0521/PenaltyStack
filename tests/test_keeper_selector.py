import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from penalty.keeper_defenses import DEFENSE_NAMES
from penalty.keeper_selector import (CONTEXT_VERSION, CONTEXT_FIELDS, ForcedDefenseKeeper,
                                     LearnedDefenseKeeper, choose_defense, context_from_state,
                                     fit_model, load_model, probabilities, save_model, seal_model, validate_model)
from penalty.keeper_vision import VisionKeeper


def state(time, crossing=.1, velocity=-1., x=1., var_y=0.):
    return {'version': 'vision_state_v2', 'time_s': time, 'capture_s': time,
            'remaining_time': 28.-time,
            'ball': {'visible': True, 'position_valid': True, 'velocity_valid': True,
                     'age_s': .01, 'position': [x, crossing], 'velocity': [velocity, 0.],
                     'velocity_covariance': [[0., 0.], [0., var_y]]}}


def core_state(observation):
    return dict(observation, version='vision_state_v1')


def rows(offset, count):
    records = []
    for index in range(count):
        crossing = -1.+2.*(index+.5)/count
        # Two useful mirrors occupy the negative/positive half of the central band;
        # all six heads still receive physical binary outcomes.
        targets = [float(abs(crossing) <= .3 and crossing >= 0),
                   float(abs(crossing) <= .3 and crossing < 0),
                   float(.3 < crossing <= .65), float(-.65 <= crossing < -.3),
                   float(crossing > .65), float(crossing < -.65)]
        records.append({'style': 'a' if index % 2 else 'b', 'seed': offset+index,
                        'context': {'version': CONTEXT_VERSION,
                                    'values': [crossing, 1.12, 1., crossing, -1., 0., .01, 0., 0.]},
                        'targets': dict(zip(DEFENSE_NAMES, targets))})
    return records


class ForbiddenState(dict):
    def __getitem__(self, key):
        if key in ('opponent', 'orientation', 'truth', 'target', 'seed', 'style', 'sensors'):
            raise AssertionError('unauthorized selector input: '+key)
        return super().__getitem__(key)


class KeeperSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.validation = rows(1000, 72), rows(2000, 48)
        cls.model = fit_model(cls.train, cls.validation,
                              regularizations=(.001, .01), margins=(0., .03), epochs=650, seed=7)

    def test_camera_context_has_only_declared_fields(self):
        observation = ForbiddenState(state(.1))
        observation.update(opponent=object(), truth=object(), target=object(), style=object(), seed=object())
        context = context_from_state(observation, {'crossing_y': .2, 'time_to_line': 1.1})
        self.assertEqual(len(CONTEXT_FIELDS), 9)
        self.assertEqual(context['values'], [.2, 1.1, 1., .1, -1., 0., .01, 0., 0.])
        keeper = LearnedDefenseKeeper(self.model)
        keeper(observation)
        later = ForbiddenState(state(.2))
        self.assertTrue(keeper(later).fires)

    def test_all_six_are_eligible_even_when_core_selected_other_side(self):
        for crossing in (-.8, .1, .8):
            for defense in DEFENSE_NAMES:
                keeper = ForcedDefenseKeeper(defense)
                self.assertFalse(keeper(state(.1, crossing)).fires)
                action = keeper(state(.2, crossing))
                self.assertTrue(action.fires)
                self.assertEqual(action.defense, defense)
                self.assertEqual(keeper.decision['core_side'], -1 if crossing < -.3 else 1 if crossing > .3 else 0)
                self.assertIsNotNone(keeper.context)

    def test_forced_timing_exactly_matches_v2_temporal_gates(self):
        rng = np.random.default_rng(19)
        for trial in range(60):
            crossing = rng.uniform(-.8, .8)
            velocity = rng.uniform(-1.5, -.25)
            core, keeper = VisionKeeper(), ForcedDefenseKeeper(DEFENSE_NAMES[trial % 6])
            for index in range(35):
                now = index*.04
                observation = state(now, crossing, velocity, max(.01, 1.8+velocity*now))
                if index % 13 == 4:
                    observation['ball']['visible'] = False
                if index % 17 == 5:
                    observation['capture_s'] = max(0., (index-1)*.04)
                expected, actual = core(core_state(observation)), keeper(observation)
                self.assertEqual(actual.trigger, expected['trigger'])
                self.assertEqual(keeper.decision['core_side'], expected['block_side'])

    def test_prediction_sticky_choice_deadline_and_reset(self):
        keeper = LearnedDefenseKeeper(self.model)
        keeper(state(.1, -.15))
        first = keeper(state(.2, -.15))
        context = copy.deepcopy(keeper.context)
        scores = copy.deepcopy(keeper.scores)
        changed = state(.3, .9)
        changed['ball']['velocity_covariance'][1][1] = float('nan')
        later = keeper(changed)
        self.assertEqual(later, first)
        self.assertEqual(keeper.context, context)
        self.assertEqual(keeper.scores, scores)
        self.assertFalse(keeper(state(28.)).fires)
        self.assertEqual(keeper.committed_defense, first.defense)
        keeper.reset()
        self.assertIsNone(keeper.context)
        self.assertFalse(keeper(state(.1)).fires)

    def test_nonfinite_context_falls_back_without_changing_core_trigger(self):
        observation = state(.1, -.15, var_y=float('nan'))
        with self.assertRaises(ValueError):
            context_from_state(observation, {'crossing_y': -.15, 'time_to_line': 1.12})
        keeper = LearnedDefenseKeeper(self.model)
        keeper(observation)
        action = keeper(state(.2, -.15, var_y=float('nan')))
        self.assertTrue(action.fires)
        self.assertEqual(action.defense, 'center')
        self.assertIsNone(keeper.context)
        self.assertIsNotNone(keeper.decision['context_error'])

    def test_camera_model_fit_learns_distinct_defenses(self):
        self.assertGreaterEqual(self.model['fit_report']['selected_validation']['macro_save_rate'], .9)
        selected = {choose_defense(self.model, row['context'], 'center')[0] for row in self.validation}
        self.assertEqual(selected, set(DEFENSE_NAMES))
        self.assertEqual(self.model['weights'].__len__(), 6)
        self.assertEqual(len(self.model['weights'][0]), 12)

    def test_train_only_scaler_and_disjoint_seed_groups(self):
        expected = np.mean([r['context']['values'] for r in self.train], axis=0)
        np.testing.assert_allclose(self.model['mean'], expected)
        leaked = copy.deepcopy(self.validation)
        leaked[0]['seed'] = self.train[0]['seed']
        leaked[0]['style'] = 'unseen_style'
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            fit_model(self.train, leaked, epochs=2)

    def test_masked_sim_errors_and_no_context_are_reported(self):
        records = copy.deepcopy(self.train)
        records[0]['targets']['left'] = None
        records.append({'style': 'c', 'seed': 9900, 'context': None, 'eligible': False,
                        'targets': dict.fromkeys(DEFENSE_NAMES, 0)})
        model = fit_model(records, self.validation, regularizations=(.01,), margins=(.03,), epochs=20)
        self.assertEqual(model['fit_report']['masked_error_targets'], 1)
        self.assertEqual(model['fit_report']['excluded_no_context'], 1)
        self.assertEqual(model['fit_report']['training_contexts'], len(self.train))

    def test_validation_errors_remain_in_denominator(self):
        validation = copy.deepcopy(self.validation)
        for row in validation:
            row['targets'] = dict.fromkeys(DEFENSE_NAMES, None)
        model = fit_model(self.train, validation, regularizations=(.01,), margins=(.03,), epochs=20)
        self.assertEqual(model['fit_report']['selected_validation']['macro_save_rate'], 0.)
        self.assertEqual(model['fit_report']['validation_rows'], len(validation))

    def test_tie_margin_keeps_original_and_inference_is_deterministic(self):
        model = copy.deepcopy(self.model)
        model['weights'] = np.zeros((6, 12)).tolist()
        model['bias'] = np.zeros(6).tolist()
        # choose_defense is an internal numerical operation; loaded policies verify fingerprints.
        context = self.validation[0]['context']
        for name in ('center', 'left', 'right'):
            selected, scores = choose_defense(model, context, name)
            self.assertEqual(selected, name)
            self.assertEqual(scores, [.5]*6)
        np.testing.assert_array_equal(probabilities(self.model, context), probabilities(self.model, context))

    def test_roundtrip_and_tampering_dimensions_versions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            save_model(self.model, path)
            self.assertEqual(load_model(path), self.model)
            original = json.loads(path.read_text())
            for key, value in (('weights', [[0.]]*6), ('version', 'unknown'),
                               ('mean', [float('nan')]*9), ('bias', [0.]*6),
                               ('defenses', list(reversed(DEFENSE_NAMES)))):
                changed = copy.deepcopy(original)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_model(changed)

    def test_policy_snapshot_is_unchanged_by_caller_model_edits(self):
        model = copy.deepcopy(self.model)
        keeper = LearnedDefenseKeeper(model)
        model['weights'][0][0] += 1000.
        self.assertEqual(keeper.model['weights'], self.model['weights'])

    def test_signed_incompatible_semantics_are_rejected(self):
        for key, value in (('feature_normalization_clip', 2.),
                           ('orientation_used', True), ('learned_joint_trajectories', True)):
            model = copy.deepcopy(self.model)
            model[key] = value
            payload = {name: item for name, item in model.items() if name != 'fingerprint'}
            encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
            model['fingerprint'] = hashlib.sha256(encoded).hexdigest()
            with self.assertRaises(ValueError):
                validate_model(model)

    def test_sealed_provenance_is_signed_and_isolated_from_caller(self):
        provenance = {'dataset_sha256': 'a'*64, 'test_used_for_fit': False}
        sealed = seal_model(self.model, provenance)
        self.assertNotEqual(sealed['fingerprint'], self.model['fingerprint'])
        self.assertNotIn('training_provenance', self.model)
        provenance['test_used_for_fit'] = True
        self.assertFalse(sealed['training_provenance']['test_used_for_fit'])
        validate_model(sealed)
        sealed['training_provenance']['test_used_for_fit'] = True
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            validate_model(sealed)

    def test_no_outcomes_or_duplicate_contexts_rejected(self):
        rows_ = copy.deepcopy(self.train)
        for row in rows_:
            row['targets']['wide_left'] = None
        with self.assertRaisesRegex(ValueError, 'each defense'):
            fit_model(rows_, self.validation, epochs=2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            fit_model(self.train+[self.train[0]], self.validation, epochs=2)


if __name__ == '__main__':
    unittest.main()
