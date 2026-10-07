"""Tactical environments over the original B-Human engine in MuJoCo."""
import unittest

from penalty.motion_contract import executable_path
from penalty.rl.contracts import KICK_RIGHT, KeeperAction, StrikerAction
from penalty.rl.evaluation import make_env, rollout
from penalty.rl.policies.baselines import ScriptedKeeperPolicy, ScriptedStrikerPolicy


@unittest.skipUnless(executable_path().is_file(), 'B-Human runtime not built')
class BHumanTacticalTests(unittest.TestCase):
    def env(self, role, opponent):
        env = make_env(role, 'bhuman', opponent)
        self.addCleanup(env.close)
        return env

    def test_scripted_striker_reproduces_scripted_penalty(self):
        env = self.env('striker', 'scripted')
        _, info = env.reset(seed=0)
        self.assertEqual(info['backend'], 'bhuman')
        self.assertIn('scene ball position', info['privileged_inputs'])
        record = rollout(env, ScriptedStrikerPolicy(), seed=0)
        # Same outcome as `python -m penalty.simulator --headless`.
        self.assertEqual(record['outcome'], 'blocked')
        self.assertEqual(record['steps'], 1)
        self.assertEqual(env.world.sim.kick_contact_count, 1)
        self.assertFalse(record['fallen'])

    def test_aimed_shot_beats_static_keeper(self):
        env = self.env('striker', 'static')
        action = StrikerAction(.4, KICK_RIGHT, .7, 1.).to_normalized()
        record = rollout(env, lambda _observation: action, seed=0)
        self.assertEqual(record['outcome'], 'goal')
        self.assertLess(record['target_error'], .15)
        self.assertEqual(env.world.striker.state, 'STOP')

    def test_scripted_keeper_saves_scripted_shot(self):
        env = self.env('keeper', 'scripted')
        record = rollout(env, ScriptedKeeperPolicy(), seed=0)
        self.assertEqual(record['outcome'], 'save')
        self.assertTrue(record['contact'])
        self.assertIsNotNone(record['reaction_time'])
        self.assertFalse(record['early_dive'])

    def test_early_lateral_dive_falls_before_the_shot(self):
        env = self.env('keeper', 'scripted')
        dive = KeeperAction(1, 1., .5).to_normalized()
        record = rollout(env, lambda _observation: dive, seed=0)
        self.assertEqual(record['outcome'], 'keeper_fall')
        self.assertTrue(record['early_dive'])
        self.assertTrue(record['fallen'])

    def test_native_failure_ends_episode_as_sim_error(self):
        env = self.env('striker', 'scripted')
        env.reset(seed=0)

        def broken():
            raise RuntimeError('striker: Nonfinite or out-of-range sensor')
        env.world.session.tick = broken
        _, reward, terminated, truncated, info = env.step(ScriptedStrikerPolicy()(None))
        self.assertEqual((info['outcome'], terminated, truncated), ('sim_error', True, False))
        self.assertAlmostEqual(reward, -10.02)
        self.assertIsNone(env.world.session)


if __name__ == '__main__':
    unittest.main()
