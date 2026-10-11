import unittest
from types import SimpleNamespace

import numpy as np

from penalty.rl.striker_calibration import (ACTION_COMPONENTS, CALIBRATION_VERSION,
    INITIALIZATION_VERSION, action_head_scales, apply_head_calibration)


class TestStrikerCalibration(unittest.TestCase):
    def test_scales_preserve_small_outputs_and_bound_measured_percentile(self):
        means = np.array([[.2, -20., 4., -30.], [.4, 20., -4., 30.]])
        result = action_head_scales(means)
        np.testing.assert_allclose(result['action_head_row_scales'], [1., .0375, .1875, .025])
        self.assertEqual(result['raw_mean_clip_fraction'], [0., 1., 1., 1.])
        self.assertEqual(result['scaled_mean_clip_fraction'], [0., 0., 0., 0.])

    def test_invalid_statistics_are_rejected(self):
        for means in ([], [[1, 2]], [[0, 0, 0, np.nan]], [[0, 0, 0, np.inf]]):
            with self.subTest(means=means), self.assertRaises(ValueError):
                action_head_scales(means)

    def test_only_head_parameters_and_their_optimizer_state_change(self):
        import torch
        torch.manual_seed(100)
        policy = torch.nn.Module()
        policy.action_net = torch.nn.Linear(3, 4)
        policy.value_net = torch.nn.Linear(3, 1)
        policy.optimizer = torch.optim.Adam(policy.parameters())
        x = torch.tensor([1., 2., 3.])
        (policy.action_net(x).sum()+policy.value_net(x).sum()).backward()
        policy.optimizer.step()
        weights = policy.action_net.weight.detach().clone()
        bias = policy.action_net.bias.detach().clone()
        value = policy.value_net.weight.detach().clone()
        value_state = policy.optimizer.state[policy.value_net.weight]['exp_avg'].clone()
        report = {'version': CALIBRATION_VERSION, 'initialization': INITIALIZATION_VERSION,
                  'validation_or_test_used': False, 'percentile_uses_only_pre_release_frames': True,
                  'action_components': list(ACTION_COMPONENTS),
                  'action_head_row_scales': [1., .1, .2, .3]}
        proof = apply_head_calibration(SimpleNamespace(policy=policy), report)
        scales = torch.tensor(report['action_head_row_scales'])
        torch.testing.assert_close(policy.action_net.weight, weights*scales[:, None])
        torch.testing.assert_close(policy.action_net.bias, bias*scales)
        torch.testing.assert_close(policy.value_net.weight, value)
        torch.testing.assert_close(policy.optimizer.state[policy.value_net.weight]['exp_avg'], value_state)
        self.assertNotIn(policy.action_net.weight, policy.optimizer.state)
        self.assertNotIn(policy.action_net.bias, policy.optimizer.state)
        self.assertTrue(proof['actor_features_and_value_unchanged'])

    def test_calibration_cannot_use_test_data_or_out_of_bounds_scales(self):
        report = {'version': CALIBRATION_VERSION, 'initialization': INITIALIZATION_VERSION,
                  'validation_or_test_used': True, 'percentile_uses_only_pre_release_frames': True,
                  'action_components': list(ACTION_COMPONENTS), 'action_head_row_scales': [1.]*4}
        with self.assertRaises(ValueError):
            apply_head_calibration(None, report)


if __name__ == '__main__':
    unittest.main()
