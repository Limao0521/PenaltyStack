"""Temporal decisions and paired benchmark accounting, without hardware."""
import math
import unittest

import numpy as np

from penalty.rl.contracts import BLOCK_CENTER, BLOCK_LEFT, BLOCK_RIGHT, FIELD, KeeperAction
from penalty.rl.robust_keeper import RobustKeeperPolicy, RobustKeeperSettings


def observation(time, y=0., vx=0., vy=0., visible=True, age=0.):
    o = np.zeros(19, np.float32)
    for name, value in dict(time_s=time, ball_x=1., ball_y=y, ball_vx=vx, ball_vy=vy,
                            ball_visible=float(visible), ball_age=age, opponent_x=1.2,
                            opponent_yaw=math.pi).items():
        o[FIELD[name]] = value
    return o


class RobustKeeperTests(unittest.TestCase):
    def action(self, policy, *args, **kwargs):
        return KeeperAction.from_normalized(policy(observation(*args, **kwargs)))

    def test_rejects_motion_away_stale_ball_and_single_spike(self):
        policy = RobustKeeperPolicy()
        self.assertFalse(self.action(policy, 0., vx=.5).fires)
        self.assertFalse(self.action(policy, .1, vx=-1., visible=False).fires)
        self.assertFalse(self.action(policy, .2, vx=-1., age=.5).fires)
        self.assertFalse(self.action(policy, .3, vx=-1.).fires)
        self.assertFalse(self.action(policy, .4, vx=0.).fires)
        self.assertFalse(self.action(policy, .5, vx=-1.).fires)
        self.assertTrue(self.action(policy, .6, vx=-1.).fires)

    def test_confirms_then_tracks_the_ball_despite_opponent_pose(self):
        for vy, expected in ((-.5, BLOCK_RIGHT), (.5, BLOCK_LEFT), (0., BLOCK_CENTER)):
            policy = RobustKeeperPolicy()
            policy(observation(0.))
            self.assertFalse(self.action(policy, .1, vx=-1., vy=vy).fires)
            action = self.action(policy, .2, vx=-1., vy=vy)
            self.assertEqual((action.block_side, action.fires), (expected, True))

    def test_slow_corner_waits_for_arrival_central_guard_does_not(self):
        policy = RobustKeeperPolicy()
        self.assertFalse(self.action(policy, .1, vx=-.3, vy=.15).fires)
        self.assertFalse(self.action(policy, .2, vx=-.3, vy=.15).fires)
        self.assertEqual(policy.decision['reason'], 'waiting_for_arrival')
        center = RobustKeeperPolicy()
        self.action(center, .1, vx=-.3)
        self.assertTrue(self.action(center, .2, vx=-.3).fires)

    def test_duplicate_frames_do_not_confirm_and_reset_removes_commitment(self):
        policy = RobustKeeperPolicy()
        self.assertFalse(self.action(policy, .1, vx=-1.).fires)
        self.assertFalse(self.action(policy, .1, vx=-1.).fires)
        self.assertTrue(self.action(policy, .2, vx=-1.).fires)
        self.assertTrue(self.action(policy, .3, vx=1.).fires)
        policy.reset()
        self.assertFalse(self.action(policy, 0., vx=-1.).fires)

    def test_invalid_observation_and_configuration_fail_safely(self):
        policy = RobustKeeperPolicy()
        for values in (np.zeros(18), np.full(19, np.nan)):
            self.assertFalse(KeeperAction.from_normalized(policy(values)).fires)
        for settings in ({'deceleration': float('nan')}, {'history': 0}, {'confirm_frames': 1.5},
                         {'corner_y': .9}, {'dive_lead_s': -1.}):
            with self.assertRaises(ValueError):
                RobustKeeperSettings(**settings)


class KeeperBenchmarkTests(unittest.TestCase):
    def test_each_style_is_seeded_independently_of_keeper(self):
        from penalty.rl.keeper_benchmark import VariedStrikerPolicy
        for style in ('center', 'mixed', 'corners', 'soft', 'late', 'feint'):
            a, b = VariedStrikerPolicy(style, 123), VariedStrikerPolicy(style, 123)
            for now in (0., 2., 5., 10.):
                np.testing.assert_array_equal(a(observation(now)), b(observation(now)))
        feint = VariedStrikerPolicy('feint', 123)
        self.assertFalse(feint(observation(0.))[3] >= 0.)
        self.assertLess(feint(observation(0.))[0]*feint(observation(5.))[0], 0.)

    def test_macro_average_and_worst_group_do_not_hide_misses(self):
        from penalty.rl.keeper_benchmark import aggregate
        summary = aggregate({'center': [{'result': 'save', 'seed': 1}],
                             'corners': [{'result': 'goal', 'seed': 1}, {'result': 'miss', 'seed': 2}]})
        self.assertEqual((summary['macro_save_rate'], summary['worst_style_save_rate']), (.5, 0.))
        self.assertEqual(summary['worst_style'], 'corners')

    def test_paired_comparison_rejects_mismatched_seeds(self):
        from penalty.rl.keeper_benchmark import paired_saves
        a = [{'result': 'save', 'seed': i} for i in range(8)]
        b = [{'result': 'goal', 'seed': i} for i in range(8)]
        result = paired_saves(a, b)
        self.assertEqual((result['only_first_saves'], result['only_second_saves']), (8, 0))
        self.assertAlmostEqual(result['p_exact'], 2/2**8)
        with self.assertRaises(ValueError):
            paired_saves(a, b[:4])

    def test_sensor_stress_delays_and_marks_age_without_altering_v1(self):
        from penalty.rl.keeper_noise import KeeperSensorStressV2
        rng = np.random.default_rng(0)
        noise = KeeperSensorStressV2(yaw_sigma=0., velocity_sigma=0., delay_steps=1)
        noise(observation(0., vx=0.), rng)
        delayed = noise(observation(.1, vx=-1.), rng)
        self.assertAlmostEqual(delayed[FIELD['ball_vx']], 0.)
        self.assertAlmostEqual(delayed[FIELD['time_s']], .1)
        self.assertAlmostEqual(delayed[FIELD['ball_age']], .1)
        self.assertEqual(noise.version, 'keeper_sensor_stress_v2')

    def test_bootstrap_clusters_the_same_seed_across_styles(self):
        from penalty.rl.keeper_benchmark import compare_styles
        saves = [{'result': 'save', 'seed': i} for i in range(8)]
        goals = [{'result': 'goal', 'seed': i} for i in range(8)]
        comparison = compare_styles({'a': saves, 'b': saves}, {'a': goals, 'b': saves})
        self.assertEqual(comparison['macro_gain'], .5)
        np.testing.assert_allclose(comparison['macro_gain_ci95'], [.5, .5])

    def test_new_policy_is_available_through_standard_interface(self):
        from penalty.rl.policies.baselines import make_policy
        self.assertIsInstance(make_policy('robust', 'keeper'), RobustKeeperPolicy)


if __name__ == '__main__':
    unittest.main()
