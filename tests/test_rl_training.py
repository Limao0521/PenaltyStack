"""PPO training plumbing: save/load of trained policies and the plan."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from penalty.rl.evaluation import evaluate

HAS_SB3 = importlib.util.find_spec('stable_baselines3') is not None


@unittest.skipUnless(HAS_SB3, 'stable-baselines3 not installed')
class TrainingTests(unittest.TestCase):
    def test_plan_shares_bhuman_hours_per_role(self):
        from penalty.rl.training import overnight_plan
        stages = overnight_plan()
        self.assertEqual(len({s.name for s in stages}), len(stages))
        bhuman = [s for s in stages if s.backend == 'bhuman']
        self.assertAlmostEqual(sum(s.share for s in bhuman), 1.)
        self.assertTrue(all(s.timesteps > 0 for s in stages if s.backend == 'synthetic'))
        self.assertEqual(stages[-1].backend, 'bhuman')

    def test_trained_keeper_round_trip(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from penalty.rl.policies.baselines import make_policy
        from penalty.rl.training import build_env, save_policy
        venv = VecNormalize(DummyVecEnv([lambda: build_env('keeper', 'synthetic', 'synthetic')]),
                            norm_reward=False)
        model = PPO('MlpPolicy', venv, n_steps=128, batch_size=64, n_epochs=1, seed=0,
                    device='cpu')
        model.learn(256)
        with tempfile.TemporaryDirectory() as directory:
            save_policy(model, venv, Path(directory), 'keeper', {'stage': 'test'})
            policy = make_policy(directory, 'keeper')
            self.assertEqual(policy.frames.maxlen, 6)
            action = policy(np.zeros(19, dtype=np.float32))
            self.assertEqual(action.shape, (3,))
            self.assertTrue(np.all(np.abs(action) <= 1))
            with self.assertRaises(ValueError):
                make_policy(directory, 'striker')
            result = evaluate('keeper', directory, episodes=5, seeds=(0,))
            self.assertEqual(result['summary']['episodes'], 5)
        venv.close()


if __name__ == '__main__':
    unittest.main()
