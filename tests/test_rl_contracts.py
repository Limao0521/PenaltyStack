"""Contracts, geometry, observations, rewards, metrics, skills and search."""
import math
import unittest

import numpy as np

from penalty.control import Observation
from penalty.optimization.objective import block_objective, kick_objective
from penalty.optimization.search import cma_es, random_search
from penalty.rl import geometry
from penalty.rl.contracts import (BLOCK_LEFT, BLOCK_RIGHT, FIELD, KICK_RIGHT, OBS_HIGH, OBS_LOW,
                                  OBSERVATION_FIELDS, KeeperAction, RLObservation, StrikerAction)
from penalty.rl.controllers import TacticalKeeperController, TacticalStrikerController, kick_allowed
from penalty.rl.metrics import summarize, wilson_interval
from penalty.rl.observation import ObservationBuilder, WorldState
from penalty.rl.rewards import StepEvents, get_reward, keeper_reward_v1, striker_reward_v1
from penalty.skills.block import BlockParameters, ParameterizedBlock
from penalty.skills.kick import KickParameters, NOMINAL_POWER, ParameterizedKick


class ContractTests(unittest.TestCase):
    def test_observation_round_trip_and_bounds(self):
        values = np.arange(len(OBSERVATION_FIELDS), dtype=np.float32) * .1
        obs = RLObservation.from_array(values)
        np.testing.assert_allclose(obs.to_array(), values)
        self.assertEqual(OBS_LOW.shape, OBS_HIGH.shape)
        self.assertTrue((OBS_LOW < OBS_HIGH).all())
        with self.assertRaises(ValueError):
            RLObservation.from_array(np.full(len(OBSERVATION_FIELDS), np.nan))

    def test_striker_action_decoding_and_validation(self):
        action = StrikerAction.from_normalized([1., .3, -1., 1.])
        self.assertAlmostEqual(action.target_y, .65)
        self.assertEqual(action.kick_side, KICK_RIGHT)
        self.assertEqual((action.kick_power, action.trigger), (0., 1.))
        self.assertTrue(action.fires)
        again = StrikerAction.from_normalized(action.to_normalized())
        self.assertEqual(again, action)
        for bad in (dict(target_y=.9), dict(kick_side=2), dict(kick_power=1.5),
                    dict(trigger=float('nan'))):
            arguments = dict(target_y=0., kick_side=0, kick_power=.5, trigger=0.)
            arguments.update(bad)
            with self.assertRaises(ValueError):
                StrikerAction(**arguments)
        with self.assertRaises(ValueError):
            StrikerAction.from_normalized([0., 0., 0.])

    def test_keeper_action_bins(self):
        self.assertEqual(KeeperAction.from_normalized([.9, 1., 0.]).block_side, BLOCK_LEFT)
        self.assertEqual(KeeperAction.from_normalized([-.9, 1., 0.]).block_side, BLOCK_RIGHT)
        self.assertFalse(KeeperAction.from_normalized([0., -1., 0.]).fires)
        for side in (-1, 0, 1):
            action = KeeperAction(side, 1., .25)
            self.assertEqual(KeeperAction.from_normalized(action.to_normalized()), action)
        with self.assertRaises(ValueError):
            KeeperAction(3, 1., .5)


class GeometryTests(unittest.TestCase):
    def test_robot_frame(self):
        # Striker on the mark facing the goal: the goal centre is straight
        # ahead and its LEFT_POST is on its left.
        origin, yaw = geometry.STRIKER_START
        self.assertAlmostEqual(geometry.bearing(geometry.GOAL_CENTER, origin, yaw), 0., places=9)
        self.assertGreater(geometry.bearing(geometry.LEFT_POST, origin, yaw), 0)
        x, y = geometry.to_robot((0., 1.), (0., 0.), math.pi/2)
        self.assertAlmostEqual(x, 1.)
        self.assertAlmostEqual(y, 0.)

    def test_predict_goal_crossing(self):
        prediction = geometry.predict_goal_crossing((1., 0.), (-2., .5))
        self.assertAlmostEqual(prediction['time_to_goal'], .5)
        self.assertAlmostEqual(prediction['predicted_goal_y'], .25)
        slowed = geometry.predict_goal_crossing((1., 0.), (-2., .5), deceleration=.6)
        self.assertGreater(slowed['time_to_goal'], .5)
        self.assertAlmostEqual(slowed['predicted_goal_y'], .25)
        self.assertIsNone(geometry.predict_goal_crossing((1., 0.), (1., 0.)))
        self.assertIsNone(geometry.predict_goal_crossing((1., 0.), (-.3, 0.), deceleration=1.))
        self.assertIsNone(geometry.predict_goal_crossing((1., 0.), (0., 0.)))


class ObservationTests(unittest.TestCase):
    def state(self, t, ball, visible=True):
        return WorldState(time_s=t, remaining_time=28-t, ball_xy=ball,
                          robot_xy=geometry.KEEPER_START[0], robot_yaw=0.,
                          opponent_xy=geometry.STRIKER_START[0], opponent_yaw=math.pi,
                          ball_visible=visible)

    def test_relative_frame_velocity_and_dropout(self):
        builder = ObservationBuilder()
        first = builder.build(self.state(0., (.1, 0.)))
        self.assertAlmostEqual(first.ball_x, 1.18)
        self.assertAlmostEqual(first.opponent_x, 1.38)
        self.assertAlmostEqual(abs(first.opponent_yaw), math.pi)
        self.assertEqual((first.ball_vx, first.ball_vy), (0., 0.))
        moving = builder.build(self.state(.1, (-.1, .05)))
        self.assertAlmostEqual(moving.ball_vx, -2.)
        self.assertAlmostEqual(moving.ball_vy, .5)
        lost = builder.build(self.state(.3, (-.5, .2), visible=False))
        self.assertEqual(lost.ball_visible, 0.)
        self.assertAlmostEqual(lost.ball_age, .2)
        self.assertAlmostEqual(lost.ball_x, moving.ball_x)


class RewardMetricTests(unittest.TestCase):
    def test_reward_tables(self):
        self.assertAlmostEqual(striker_reward_v1(StepEvents()), -.02)
        self.assertAlmostEqual(striker_reward_v1(StepEvents('goal')), 9.98)
        self.assertAlmostEqual(striker_reward_v1(StepEvents('blocked')), -8.02)
        self.assertAlmostEqual(striker_reward_v1(StepEvents('fall', fallen=True)), -10.02)
        self.assertAlmostEqual(keeper_reward_v1(StepEvents('save', fallen=True, contact=True)), 13.98)
        self.assertAlmostEqual(keeper_reward_v1(StepEvents('goal', fallen=True)), -15.02)
        self.assertAlmostEqual(keeper_reward_v1(StepEvents(early_dive=True)), -2.02)
        self.assertIs(get_reward('keeper_reward_v1'), keeper_reward_v1)
        with self.assertRaises(ValueError):
            get_reward('striker_reward_v0')

    def test_summary(self):
        records = [{'outcome': 'goal', 'ball_speed': 1., 'fallen': False},
                   {'outcome': 'blocked', 'ball_speed': 3., 'fallen': True}]
        summary = summarize(records, 'striker')
        self.assertEqual(summary['rates']['goal_rate'], .5)
        self.assertEqual(summary['rates']['any_fall_rate'], .5)
        self.assertEqual(summary['secondary']['ball_speed']['mean'], 2.)
        low, high = wilson_interval(5, 10)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)


class ControllerTests(unittest.TestCase):
    def ready(self, now):
        return Observation(now, start=True, ball_range=.2, ball_bearing=0., goal_bearing=0.,
                           motion_busy=False, motion_done=False)

    def test_striker_holds_select_shot_until_fire(self):
        controller = TacticalStrikerController()
        self.assertEqual(controller.update(self.ready(0.)).state, 'SELECT_SHOT')
        self.assertFalse(controller.kick_released)
        self.assertEqual(controller.last_rejection, 'waiting')
        controller.fire = True
        command = controller.update(self.ready(.1))
        self.assertTrue(command.kick)
        self.assertTrue(controller.kick_released)
        self.assertEqual(controller.update(self.ready(.2)).state, 'KICK')

    def test_safety_filter(self):
        self.assertEqual(kick_allowed(.2, 0., False, 0., 0.), (True, 'ok'))
        self.assertEqual(kick_allowed(.4, 0., False, 0., 0.)[1], 'ball_out_of_range')
        self.assertEqual(kick_allowed(.2, .3, False, 0., 0.)[1], 'ball_bearing')
        self.assertEqual(kick_allowed(.2, 0., True, 0., 0.)[1], 'motion_busy')
        self.assertEqual(kick_allowed(.2, 0., False, .5, 0.)[1], 'unstable')
        self.assertEqual(kick_allowed(float('nan'), 0., False, 0., 0.)[1], 'non_finite')
        unstable = TacticalStrikerController(stability=lambda: (.5, 0.))
        unstable.fire = True
        self.assertFalse(unstable.update(self.ready(0.)).kick)
        self.assertEqual(unstable.last_rejection, 'unstable')

    def test_aim_offset_moves_to_kick_pose(self):
        controller = TacticalStrikerController()
        controller.aim_offset = .3
        command = controller.update(self.ready(0.))
        self.assertEqual(command.state, 'ALIGN_TO_SHOT')
        self.assertGreater(command.turn, 0)
        self.assertLess(command.lateral, 0)  # orbit: step right to aim left

    def test_keeper_commits_once(self):
        controller = TacticalKeeperController()
        seen = Observation(0., start=True, ball_range=1., ball_bearing=.1, goal_bearing=0.)
        command = controller.update(seen)
        self.assertEqual((command.state, command.block_now), ('TRACK_BALL', False))
        controller.commit('left')
        controller.commit('right')
        command = controller.update(seen)
        self.assertEqual((command.state, command.block, command.block_now), ('BLOCK', 'left', True))


class SkillTests(unittest.TestCase):
    def test_kick_bounds_and_runtime_support(self):
        kick = ParameterizedKick(KICK_RIGHT)
        self.assertEqual(kick.bounds(), {'power': (0., 1.)})
        request = kick.apply(KickParameters(power=NOMINAL_POWER))
        self.assertEqual(request['kick_type'], 'forwardFastRight')
        self.assertAlmostEqual(request['kick_length_mm'], 2500.)
        self.assertAlmostEqual(kick.apply(KickParameters(power=1.))['kick_length_mm'], 2900.)
        for bad in (KickParameters(power=1.2), KickParameters(power=float('nan')),
                    KickParameters(hip_rotation=.1), KickParameters(swing_amplitude=2.)):
            with self.assertRaises(ValueError):
                kick.apply(bad)
        self.assertEqual(kick.record()['runtime_supported'], ['power'])
        with self.assertRaises(ValueError):
            ParameterizedKick(5)

    def test_block_parameters_are_nominal_only(self):
        block = ParameterizedBlock(BLOCK_LEFT)
        self.assertEqual(block.bounds(), {})
        self.assertEqual(block.apply(BlockParameters()), {'block_side': 'left'})
        with self.assertRaises(ValueError):
            block.apply(BlockParameters(torso_lean=.1))
        with self.assertRaises(TypeError):
            block.apply(KickParameters())


class SearchTests(unittest.TestCase):
    @staticmethod
    def objective(x):
        return -float(np.sum((np.asarray(x) - np.array([.3, -.2, .8]))**2))

    def test_random_search_respects_bounds(self):
        bounds = [(0, 1), (-1, 0), (.5, 1)]
        result = random_search(self.objective, bounds, 50, seed=3)
        self.assertEqual(result['evaluations'], 50)
        for item in result['history']:
            for value, (low, high) in zip(item['x'], bounds):
                self.assertTrue(low <= value <= high)
        self.assertEqual(random_search(self.objective, bounds, 50, seed=3)['best_x'], result['best_x'])

    def test_cma_es_converges_inside_bounds(self):
        bounds = [(0, 1), (-1, 0), (.5, 1)]
        result = cma_es(self.objective, bounds, 300, seed=1)
        np.testing.assert_allclose(result['best_x'], [.3, -.2, .8], atol=.05)
        for item in result['history']:
            for value, (low, high) in zip(item['x'], bounds):
                self.assertTrue(low <= value <= high)
        with self.assertRaises(ValueError):
            cma_es(self.objective, [(1, 0)], 10)

    def test_objectives(self):
        good = [{'outcome': 'goal', 'ball_speed': 1., 'target_error': 0., 'kick_time': 1.}]
        bad = [{'outcome': 'blocked', 'ball_speed': 1., 'target_error': None, 'fallen': True,
                'kick_time': 1.}]
        self.assertGreater(kick_objective(good), kick_objective(bad))
        save = [{'outcome': 'save', 'contact': True, 'reaction_time': .3}]
        goal = [{'outcome': 'goal', 'fallen': True, 'reaction_time': .3, 'wrong_side': True}]
        self.assertGreater(block_objective(save), block_objective(goal))


if __name__ == '__main__':
    unittest.main()
