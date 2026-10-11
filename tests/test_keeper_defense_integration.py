import unittest
from types import SimpleNamespace
from unittest.mock import patch

from penalty.keeper_defenses import DEFENSE_NAMES, KeeperDefenseAction, catalog
from penalty.rl.contracts import KeeperAction, StrikerAction
from penalty.rl.defense_benchmark import (DefenseCameraKeeperController,
                                         SensorDefenseDuelWorld, benchmark, legacy_side)


class DefenseActionTests(unittest.TestCase):
    def test_new_action_rejects_preparation_unknown_and_nonfinite_trigger(self):
        for defense in ('prepare', 'unknown', 3):
            with self.assertRaises(ValueError):
                KeeperDefenseAction(defense, 1.)
        for trigger in (float('nan'), float('inf'), -.1, 1.1, True, '1'):
            with self.assertRaises(ValueError):
                KeeperDefenseAction('center', trigger)
        self.assertFalse(KeeperDefenseAction('wide_right', .49).fires)
        self.assertTrue(KeeperDefenseAction('wide_right', .5).fires)

    def test_preparation_is_separate_and_legacy_action_unchanged(self):
        self.assertEqual(len(catalog()['defenses']), 6)
        self.assertEqual(tuple(catalog()['preparations']), ('prepare',))
        for side in (-1, 0, 1):
            original = KeeperAction(side, 1., .5)
            self.assertEqual(KeeperAction.from_normalized(original.to_normalized()), original)


class DefenseWorldTests(unittest.TestCase):
    def test_controller_commits_selected_primitive_and_cannot_change_mid_guard(self):
        for defense in DEFENSE_NAMES:
            controller = DefenseCameraKeeperController()
            controller.requested_defense = defense
            zone = {0: 'center', 1: 'left', -1: 'right'}[legacy_side(defense)]
            controller.commit(zone)
            self.assertEqual(controller.block, defense)
            controller.requested_defense = 'center_right'
            controller.commit('center')
            self.assertEqual(controller.block, defense)
        with self.assertRaises(ValueError):
            DefenseCameraKeeperController().commit('center')

    def test_world_adapter_reaches_all_six_without_modifying_old_duel_world(self):
        world = SensorDefenseDuelWorld()
        world.keeper = DefenseCameraKeeperController()
        striker = StrikerAction(0., 0, .5, 0.)
        for defense in DEFENSE_NAMES:
            with patch('penalty.sim_orientation.SensorCameraDuelWorld.duel_step', return_value='ok') as call:
                action = KeeperDefenseAction(defense, 1.)
                self.assertEqual(world.duel_step(striker, action), 'ok')
                call.assert_called_once_with(striker, KeeperAction(legacy_side(defense), 1., .5))
                self.assertEqual(world.keeper.requested_defense, defense)
        with self.assertRaises(TypeError):
            world.duel_step(striker, KeeperAction(0, 1., .5))

    def test_benchmark_pairs_seeds_and_reports_named_defenses(self):
        def job(spec):
            return spec['style'], [(mode, {'seed': seed, 'result': 'save', 'commit_defense': 'center',
                                          'commit_side': 0, 'early_dive': False})
                                   for seed in spec['seeds'] for mode in spec['modes']]
        with patch('penalty.rl.defense_benchmark._job', side_effect=job):
            result = benchmark(('original', 'balanced'), episodes=3, seed=72,
                               workers=1, styles=('center', 'mixed'), progress=None)
        self.assertEqual(result['action_version'], 'keeper_action_v2')
        self.assertEqual(result['comparisons']['balanced_vs_original']['macro_gain'], 0.)
        self.assertEqual(result['defense_counts']['balanced'], {'center': 6})
        for style in result['styles']:
            a, b = (result['records'][mode][style] for mode in ('original', 'balanced'))
            self.assertEqual([row['seed'] for row in a], [row['seed'] for row in b])


if __name__ == '__main__':
    unittest.main()
