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

    def test_commit_after_shot_waits_for_the_kick(self):
        env = make_env('keeper', 'bhuman', 'scripted', {'commit_after_shot': True})
        self.addCleanup(env.close)
        env.reset(seed=0)
        _, _, terminated, _, info = env.step(KeeperAction(1, 1., .5).to_normalized())
        self.assertEqual(info['safety'], 'waiting_for_shot')
        self.assertFalse(terminated)
        record = rollout(env, lambda _observation: KeeperAction(1, 1., .5).to_normalized(), seed=0)
        self.assertNotEqual(record['outcome'], 'keeper_fall')
        self.assertFalse(record['early_dive'])
        self.assertIsNotNone(record['reaction_time'])

    def test_duel_scripted_players_and_keeper_gate(self):
        from penalty.rl.duel import RESULTS, DuelWorld, duel_episode, make_player, summarize_duels
        striker = make_player({'kind': 'baseline', 'name': 'scripted'}, 'striker')
        keeper = make_player({'kind': 'baseline', 'name': 'scripted'}, 'keeper')
        world = DuelWorld(seed=0)
        self.addCleanup(world.close)
        record = duel_episode(world, striker, keeper, seed=0, noise=False)
        # Same decisions as the shared FSM penalty: centred shot, blocked.
        self.assertEqual(record['result'], 'save')
        self.assertGreater(record['contact_time'], 0)
        self.assertEqual(summarize_duels([record])['rates']['save_rate'], 1.)
        dive = lambda _observation: KeeperAction(1, 1., .5).to_normalized()  # noqa: E731
        self.assertEqual(duel_episode(world, striker, dive, seed=0, noise=False)['result'], 'keeper_fall')
        gated = DuelWorld(seed=0, keeper_gate=True)
        self.addCleanup(gated.close)
        record = duel_episode(gated, striker, dive, seed=0, noise=False)
        self.assertIn(record['result'], RESULTS)
        self.assertNotEqual(record['result'], 'keeper_fall')
        self.assertGreaterEqual(record['commit_time'], record['contact_time'])

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

    def test_duel_preserves_native_failure_reason_and_collects_children(self):
        from penalty.rl.duel import DuelWorld
        import numpy as np
        world = DuelWorld()
        self.addCleanup(world.close)
        world.reset(np.random.default_rng(0))
        children = [bridge.process for bridge in world.sim.bridges.values()]

        def broken():
            raise RuntimeError('striker: Nonfinite or out-of-range sensor')
        world._step_window = broken
        result = world.duel_step(StrikerAction(0., 0, .7, 1.), KeeperAction(0, 0., .5))
        self.assertEqual(result[2], 'sim_error')
        self.assertIn('out-of-range sensor', world.record()['sim_error'])
        self.assertIsNone(world.session)
        self.assertTrue(all(process.poll() is not None for process in children))


if __name__ == '__main__':
    unittest.main()
