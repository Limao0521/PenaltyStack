import json
from pathlib import Path
import shutil
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from penalty.rl.actors import NumpyPolicy
from penalty.rl.async_ppo import export_policy
from penalty.rl.striker_preserving_policy import (
    INITIAL_STD, MAX_STD, MIN_STD, TAIL_SLOPE, PreservingActorCriticPolicy,
    actor_parameters_sha256, clamp_std, compress_action_tails, initialize_preserving_model)


BASE = Path(__file__).resolve().parents[1]/'models/rl/striker_v2'


class PlaceholderEnv(gym.Env):
    def __init__(self, model):
        self.observation_space = model.observation_space
        self.action_space = model.action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(19, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(19, dtype=np.float32), 0., True, False, {}


class TestTailCompression(unittest.TestCase):
    def test_clipped_means_are_exactly_preserved_including_large_tails(self):
        for dtype in (torch.float32, torch.float64):
            limit = torch.finfo(dtype).max
            means = torch.tensor([-limit, -30., -1.01, -1., -.5, 0., .5, 1., 1.01, 30., limit],
                                 dtype=dtype)
            actual = compress_action_tails(means)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.equal(actual.clamp(-1., 1.), means.clamp(-1., 1.)))
            self.assertTrue(torch.equal(actual[3:8], means[3:8]))

    def test_gradients_are_the_true_piecewise_derivative(self):
        means = torch.tensor([-30., -1.01, -.5, 0., .5, 1.01, 30.], dtype=torch.float64,
                             requires_grad=True)
        compress_action_tails(means).sum().backward()
        torch.testing.assert_close(means.grad,
                                   torch.tensor([TAIL_SLOPE, TAIL_SLOPE, 1., 1., 1.,
                                                 TAIL_SLOPE, TAIL_SLOPE], dtype=torch.float64))
        self.assertTrue(torch.autograd.gradcheck(compress_action_tails,
                                                (means.detach().requires_grad_(True),)))

    def test_integer_means_are_rejected(self):
        with self.assertRaises(TypeError):
            compress_action_tails(torch.tensor([1, 2]))

    def test_other_distributions_and_action_contracts_are_rejected(self):
        observation = gym.spaces.Box(-1., 1., shape=(19,), dtype=np.float32)
        for action, kwargs in ((gym.spaces.Discrete(4), {}),
                               (gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float32), {}),
                               (gym.spaces.Box(-2., 2., shape=(4,), dtype=np.float32), {}),
                               (gym.spaces.Box(-1., 1., shape=(4,), dtype=np.float32), {'use_sde': True})):
            with self.subTest(action=action, kwargs=kwargs), self.assertRaises(ValueError):
                PreservingActorCriticPolicy(observation, action, lambda _: .0001, **kwargs)


class TestPreservingInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base = PPO.load(BASE/'model.zip', device='cpu')

    def initialize(self, **kwargs):
        return initialize_preserving_model(BASE, PlaceholderEnv(self.base), learning_rate=1e-4,
                                           n_steps=8, batch_size=8, seed=94002, **kwargs)

    def test_actor_is_byte_identical_and_only_value_head_and_sigma_change(self):
        model, proof = self.initialize()
        self.assertEqual(actor_parameters_sha256(model.policy), actor_parameters_sha256(self.base.policy))
        for name, value in self.base.policy.state_dict().items():
            if name not in ('log_std', 'value_net.weight', 'value_net.bias'):
                self.assertTrue(torch.equal(value, model.policy.state_dict()[name]), name)
        self.assertEqual(model.num_timesteps, self.base.num_timesteps)
        self.assertTrue(proof['actor_parameters_byte_identical'])
        self.assertEqual(proof['optimizer_initial_state_entries'], 0)
        self.assertFalse(proof['action_head_rescaled'])
        self.assertEqual(len(model.policy.optimizer.state), 0)
        self.assertFalse(model.policy.value_net.weight.any())
        self.assertFalse(model.policy.value_net.bias.any())
        torch.testing.assert_close(model.policy.log_std.exp(), torch.full((4,), INITIAL_STD))

    def test_initial_torch_actions_are_exactly_v2_for_normalized_observations(self):
        model, _ = self.initialize()
        observations = np.random.default_rng(94002).uniform(-10., 10., (1024, 19)).astype(np.float32)
        old_actions, _ = self.base.predict(observations, deterministic=True)
        new_actions, _ = model.predict(observations, deterministic=True)
        np.testing.assert_array_equal(new_actions, old_actions)
        # Raw saturated heads move near the action range while preserving clip.
        with torch.no_grad():
            tensor = torch.from_numpy(observations)
            latent, _ = model.policy.mlp_extractor(model.policy.extract_features(tensor))
            raw = model.policy.action_net(latent)
            mean = model.policy.get_distribution(tensor).distribution.mean
            torch.testing.assert_close(mean, compress_action_tails(raw))
            self.assertGreater(int((raw.abs() > 2.).sum()), 0)
            self.assertLess(float(mean.abs().max()), float(raw.abs().max()))

    def test_saved_model_and_existing_numpy_export_preserve_deterministic_actions(self):
        from penalty.rl.duel import NumpyTrainedPolicy, export_trained
        model, _ = self.initialize()
        observations = np.random.default_rng(95002).uniform(-10., 10., (128, 19)).astype(np.float32)
        # NumPy evaluates in float64, so parity allows ordinary float32 rounding.
        np_policy = NumpyPolicy(export_policy(model.policy))
        torch_actions, _ = model.predict(observations, deterministic=True)
        numpy_actions = np.array([np.clip(np_policy.mean(obs), -1., 1.) for obs in observations],
                                dtype=np.float32)
        np.testing.assert_allclose(numpy_actions, torch_actions, atol=2e-5, rtol=2e-5)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model.save(directory/'model.zip')
            shutil.copyfile(BASE/'obs_norm.npz', directory/'obs_norm.npz')
            (directory/'config.json').write_text(json.dumps({'role': 'striker', 'history': 1}),
                                                encoding='utf-8')
            loaded = PPO.load(directory/'model.zip', device='cpu')
            self.assertIsInstance(loaded.policy, PreservingActorCriticPolicy)
            loaded_actions, _ = loaded.predict(observations, deterministic=True)
            np.testing.assert_array_equal(loaded_actions, torch_actions)
            spec = export_trained(directory)
            player = NumpyTrainedPolicy(spec)
            base_player = NumpyTrainedPolicy(export_trained(BASE))
            raw_observations = np.random.default_rng(95003).uniform(-3., 3., (64, 19)).astype(np.float32)
            for obs in raw_observations:
                np.testing.assert_array_equal(player(obs), base_player(obs))

    def test_optimizer_is_saved_normally_after_actual_update(self):
        model, _ = self.initialize()
        model.learn(total_timesteps=8, reset_num_timesteps=False)
        self.assertGreater(len(model.policy.optimizer.state), 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.zip'
            model.save(path)
            loaded = PPO.load(path, device='cpu')
            before = model.policy.optimizer.state_dict()
            after = loaded.policy.optimizer.state_dict()
            self.assertEqual(before['param_groups'], after['param_groups'])
            self.assertEqual(before['state'].keys(), after['state'].keys())
            for key, state in before['state'].items():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(after['state'][key][name], value)
                    else:
                        self.assertEqual(after['state'][key][name], value)

    def test_sigma_clamp_and_incompatible_initialization_settings(self):
        model, _ = self.initialize()
        with torch.no_grad():
            model.policy.log_std.copy_(torch.tensor([-20., -3., 0., 20.]))
        clamp_std(model)
        expected = torch.tensor([MIN_STD, MIN_STD, MAX_STD, MAX_STD])
        torch.testing.assert_close(model.policy.log_std.exp(), expected)
        for bounds in ((0., 1.), (1., .5), (np.nan, 1.), (.1, np.inf)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                clamp_std(model, *bounds)
        with self.assertRaises(ValueError):
            self.initialize(gamma=.99)
        with self.assertRaises(ValueError):
            self.initialize(use_sde=True)
        with self.assertRaises(ValueError):
            self.initialize(policy_kwargs={})


if __name__ == '__main__':
    unittest.main()
