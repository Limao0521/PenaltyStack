"""The expanded motor library preserves v2 prediction and trigger timing."""
import math
import random
import unittest

from penalty.keeper_defenses import KeeperDefenseAction
from penalty.keeper_library import MODES, VisionLibraryKeeper
from penalty.keeper_vision import VisionKeeper


def state(t, x=1., y=0., vx=-1., vy=0., capture=None):
    return {'version': 'vision_state_v2', 'time_s': t,
            'capture_s': t if capture is None else capture, 'remaining_time': 28.-t,
            'ball': {'position': (x, y), 'velocity': (vx, vy),
                     'position_valid': True, 'velocity_valid': True, 'visible': True,
                     'age_s': 0., 'velocity_covariance': ((0., 0.), (0., 0.))}}


class VisionLibraryKeeperTests(unittest.TestCase):
    def assert_same_core_action(self, library, core, packet):
        expected = core(dict(packet, version='vision_state_v1'))
        actual = library(packet)
        self.assertIsInstance(actual, KeeperDefenseAction)
        self.assertEqual(actual.trigger, expected['trigger'])
        self.assertEqual(library.decision['core_side'], expected['block_side'])
        if expected['block_side'] or library.mode not in ('wide_left', 'wide_right'):
            self.assertEqual(actual.side, expected['block_side'])
        for name in ('reason', 'crossing_y', 'time_to_line', 'side'):
            self.assertEqual(library.decision.get(name), core.decision.get(name))
        return actual

    def test_all_modes_preserve_side_and_timing_for_left_right_and_center(self):
        for mode in MODES:
            for vy in (-.5, 0., .5):
                with self.subTest(mode=mode, vy=vy):
                    library, core = VisionLibraryKeeper(mode), VisionKeeper()
                    packets = [state(.1, vy=vy), state(.15, vy=vy, capture=.1),
                               state(.2, vy=vy), state(.3, vy=-vy), state(28., vy=vy)]
                    for packet in packets:
                        self.assert_same_core_action(library, core, packet)

    def test_long_arrival_waits_for_same_v2_threshold_then_jumps(self):
        for mode in MODES:
            library, core = VisionLibraryKeeper(mode), VisionKeeper()
            results = []
            for t, x in ((.1, 2.), (.2, 1.9), (.3, 1.8), (.4, 1.6), (.5, 1.4)):
                results.append(self.assert_same_core_action(library, core, state(t, x=x, y=.5)).fires)
            self.assertEqual(results, [False, False, False, False, True])
            self.assertEqual(library.committed_defense, 'left')

    def test_center_variants_replace_only_original_center_defense(self):
        expected = {'original': 'center', 'center_right': 'center_right',
                    'wide_left': 'wide_left', 'wide_right': 'wide_right', 'balanced': 'center'}
        for mode in MODES:
            policy = VisionLibraryKeeper(mode)
            self.assertFalse(policy(state(.1)).fires)
            action = policy(state(.2))
            self.assertTrue(action.fires)
            self.assertEqual(action.defense, expected[mode])
            self.assertEqual(policy.decision['core_side'], 0)

    def test_center_commit_remains_early_and_corner_boundaries_are_unchanged(self):
        for mode in MODES:
            for y, side in ((-.300001, -1), (-.3, 0), (.3, 0), (.300001, 1)):
                library, core = VisionLibraryKeeper(mode), VisionKeeper()
                self.assert_same_core_action(library, core, state(.1, x=3., y=y))
                action = self.assert_same_core_action(library, core, state(.2, x=3., y=y))
                self.assertEqual(library.decision['side'], side)
                # v2 commits centrally after confirmation, even when the
                # projected arrival remains more than the lateral dive lead.
                self.assertEqual(action.fires, side == 0)
                self.assertGreater(library.decision['time_to_line'], 1.8)

    def test_balanced_uses_crossing_sign_and_remains_sticky(self):
        for y, defense in ((-.2, 'center_right'), (0., 'center'), (.2, 'center')):
            policy = VisionLibraryKeeper('balanced')
            policy(state(.1, y=y))
            self.assertEqual(policy(state(.2, y=y)).defense, defense)
            self.assertEqual(policy(state(.3, y=-y, vy=.8)).defense, defense)
            self.assertEqual(policy.decision['selected_defense'], defense)

    def test_reset_and_clock_rewind_allow_a_new_selection(self):
        policy = VisionLibraryKeeper('balanced')
        for t in (.1, .2):
            policy(state(t, y=-.2))
        self.assertEqual(policy.committed_defense, 'center_right')
        self.assertFalse(policy(state(.01, y=.2)).fires)
        self.assertEqual(policy(state(.02, y=.2)).defense, 'center')
        policy.reset()
        self.assertIsNone(policy.committed_defense)
        self.assertFalse(policy(state(.1, y=-.2)).fires)

    def test_loss_restarts_confirmation_and_deadline_disables_commit(self):
        for mode in MODES:
            policy, core = VisionLibraryKeeper(mode), VisionKeeper()
            loss = state(.2)
            loss['ball']['visible'] = False
            for packet in (state(.1), loss, state(.3), state(.4), state(28.)):
                self.assert_same_core_action(policy, core, packet)

    def test_uncertain_stale_and_missing_measurements_cannot_start_a_block(self):
        cases = []
        for field in ('position_valid', 'velocity_valid', 'visible'):
            packet = state(.1)
            packet['ball'][field] = False
            cases.append(packet)
        packet = state(.1)
        packet['ball']['age_s'] = .26
        cases.append(packet)
        packet = state(.1, vx=-.2)
        packet['ball']['velocity_covariance'] = ((.1, 0.), (0., .1))
        cases.append(packet)
        for packet in cases:
            library, core = VisionLibraryKeeper('wide_left'), VisionKeeper()
            self.assertFalse(self.assert_same_core_action(library, core, packet).fires)

    def test_opponent_yaw_and_truth_are_not_read(self):
        class Inaccessible:
            def __getitem__(self, key):
                raise AssertionError('opponent or ideal data read')
            def __bool__(self):
                raise AssertionError('opponent or ideal data used')
        policy = VisionLibraryKeeper('balanced')
        for t in (.1, .2):
            packet = state(t, y=-.2)
            packet.update(opponent=Inaccessible(), truth=Inaccessible(), target_y=Inaccessible(),
                          contact=Inaccessible(), ideal=Inaccessible())
            action = policy(packet)
        self.assertEqual(action.defense, 'center_right')
        self.assertFalse(policy.decision['pose_prior_used'])

    def test_malformed_observation_waits_and_cannot_execute(self):
        for packet in ({}, None, {'version': 'vision_state_v2'},
                       dict(state(.1), ball={}), dict(state(.1), ball=None)):
            policy = VisionLibraryKeeper('wide_right')
            action = policy(packet)
            self.assertFalse(action.fires)
            self.assertIn(policy.decision['reason'], ('invalid_version', 'invalid_observation'))
        policy = VisionLibraryKeeper()
        for value in (math.nan, math.inf):
            self.assertFalse(policy(dict(state(.1), time_s=value)).fires)

    def test_wrong_state_version_is_rejected(self):
        policy = VisionLibraryKeeper()
        action = policy(dict(state(.1), version='vision_state_v1'))
        self.assertFalse(action.fires)
        self.assertEqual(policy.decision['reason'], 'invalid_version')

    def test_random_ball_tracks_match_v2_in_every_mode(self):
        # Mixed realistic captures exercise rest, uncertain velocity, gaps,
        # repeated frames, lateral timing and a final committed trajectory.
        randomizer = random.Random(80401)
        for mode in MODES:
            for _ in range(20):
                policy, core = VisionLibraryKeeper(mode), VisionKeeper()
                for index in range(12):
                    t = .1*(index+1)
                    packet = state(t, x=1.7-.07*index, y=randomizer.uniform(-.15, .15),
                                   vx=randomizer.uniform(-1.2, .1), vy=randomizer.uniform(-.4, .4))
                    packet['ball']['visible'] = randomizer.random() > .1
                    packet['ball']['velocity_valid'] = randomizer.random() > .1
                    if index % 4 == 1:
                        packet['capture_s'] = t-.1
                    self.assert_same_core_action(policy, core, packet)

    def test_description_identifies_versioned_experimental_action_and_core(self):
        description = VisionLibraryKeeper('balanced', corner_y=.31).describe()
        self.assertEqual(description['policy'], 'vision_keeper_v6')
        self.assertEqual(description['core_policy'], 'vision_keeper_v2')
        self.assertEqual(description['action_version'], 'keeper_action_v2')
        self.assertEqual(len(description['defenses']), 6)
        self.assertEqual(description['settings']['corner_y'], .31)
        self.assertFalse(description['orientation_used'])
        self.assertTrue(description['experimental'])
        with self.assertRaises(ValueError):
            VisionLibraryKeeper('adaptive_unvalidated')


if __name__ == '__main__':
    unittest.main()
