"""Regression checks for the undiscounted goal objective and failure evidence."""
import unittest

import numpy as np

from penalty.rl.striker_visual_env import StrikerVisualKeeperEnv
from penalty.rl.striker_visual_env_v2 import (
    SPLIT_BASES, StrikerVisualKeeperEnvV2, striker_visual_reward_v3,
)
from tests.test_striker_visual_env import FakeCamera, FakeWorld


class GoalObjectiveTests(unittest.TestCase):
    def env(self, cls=StrikerVisualKeeperEnvV2, **kwargs):
        env = cls(world_factory=FakeWorld, camera_factory=FakeCamera, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_long_goal_and_short_goal_have_same_undiscounted_return(self):
        for length in (1, 36, 126, 215, 280):
            for result, expected in (('goal', 1.), ('save', -1.), ('timeout', -1.)):
                env = self.env()
                env.reset(seed=104001000000)
                env.world.results = [None]*(length-1)+[result]
                rewards = []
                for _ in range(length):
                    _, reward, terminal, truncated, info = env.step(np.array([1., -1., 1., 1.]))
                    rewards.append(reward)
                self.assertTrue(terminal)
                self.assertFalse(truncated)
                self.assertEqual(sum(rewards), expected)
                self.assertEqual(info['episode_record']['return'], expected)

    def test_physical_commands_and_observations_unchanged_from_v1(self):
        old, new = self.env(StrikerVisualKeeperEnv), self.env()
        old_obs, _ = old.reset(seed=104001000000)
        new_obs, _ = new.reset(seed=104001000000)
        np.testing.assert_array_equal(old_obs, new_obs)
        for action in (np.array([1., -1., .6, 1.]), np.array([-1., 1., .8, 1.])):
            a, *_, ai = old.step(action)
            b, *_, bi = new.step(action)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(ai['result'], bi['result'])
        self.assertEqual(old.world.calls, new.world.calls)
        self.assertEqual(ai['episode_record']['first_trigger'], bi['episode_record']['first_trigger'])

    def test_new_seed_spaces_and_metadata_preserve_published_contracts(self):
        seeds = []
        for split, base in SPLIT_BASES.items():
            env = self.env(split=split)
            _, info = env.reset()
            seeds.append(env.episode_seed)
            self.assertEqual(env.episode_seed, base*1000000)
            self.assertEqual(info['environment'], 'striker_visual_keeper_env_v2')
            self.assertEqual(info['reward'], 'striker_visual_reward_v3')
            self.assertEqual(info['observation'], 'rl_obs_v1')
            self.assertEqual(info['striker_action'], 'striker_action_v1')
            self.assertEqual(info['required_training_gamma'], 1.)
            self.assertFalse(info['sim_errors_optimizable'])
        self.assertEqual(len(set(seeds)), 3)

    def test_physics_error_preserves_seed_cause_trace_and_collects_children(self):
        env = self.env()
        env.reset(seed=104001000001)
        children = env._processes()
        env.world.raise_step = RuntimeError('diagnostic native channel failure')
        _, reward, terminal, truncated, info = env.step(np.zeros(4))
        self.assertEqual((reward, terminal, truncated), (0., True, False))
        record = info['episode_record']
        self.assertFalse(record['optimizable'])
        self.assertEqual(record['seed'], 104001000001)
        self.assertEqual(record['sim_error_type'], 'RuntimeError')
        self.assertIn('diagnostic native channel failure', record['sim_error'])
        self.assertIn('raise self.raise_step', record['sim_error_traceback'])
        self.assertTrue(record['children_collected'])
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_camera_error_preserves_origin_and_collects_children(self):
        env = self.env()
        env.reset(seed=104001000002)
        children = env._processes()
        env.camera.failure = ValueError('RGB detector invalid output')
        _, _, _, _, info = env.step(np.zeros(4))
        record = info['episode_record']
        self.assertIn('RGB detector invalid output', record['sim_error_traceback'])
        self.assertEqual(record['sim_error_type'], 'ValueError')
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_backend_returned_error_is_labelled_without_inventing_original_trace(self):
        env = self.env()
        env.reset(seed=104001000003)
        children = env._processes()
        env.world.duel_step = lambda *_: (None, None, 'sim_error', {'sim_error': 'backend IPC closed'})
        _, reward, terminal, _, info = env.step(np.zeros(4))
        record = info['episode_record']
        self.assertEqual((reward, terminal), (0., True))
        self.assertEqual(record['sim_error'], 'backend IPC closed')
        self.assertEqual(record['sim_error_origin'], 'world.duel_step returned sim_error')
        self.assertFalse(record['sim_error_traceback_available'])
        self.assertNotIn('sim_error_traceback', record)
        self.assertIn('sim_error_call_site', record)
        self.assertFalse(record['optimizable'])
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_reward_rejects_unknown_outcomes_and_error_is_not_a_tactical_penalty(self):
        self.assertEqual(striker_visual_reward_v3('sim_error'), 0.)
        with self.assertRaises(ValueError):
            striker_visual_reward_v3('invalid')


if __name__ == '__main__':
    unittest.main()
