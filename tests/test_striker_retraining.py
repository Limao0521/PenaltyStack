"""Fine-tuning identity/selection and real VecEnv protocol, without physics."""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from penalty.rl import striker_retraining as training
from penalty.rl.striker_calibration import CALIBRATION_BASE, CALIBRATION_VERSION
from penalty.rl.contracts import OBS_LOW, OBS_HIGH


class FakePhysicalEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, index=0, close_path=None, fail=False):
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
        self.index, self.ordinal, self.steps = index, 0, 0
        self.close_path, self.fail = close_path, fail

    def reset(self, *, seed=None, options=None):
        self.ordinal += 1
        self.steps = 0
        self.episode_seed = training.episode_seed('train', self.ordinal*2+self.index)
        return np.zeros(19, np.float32), {'episode_seed': self.episode_seed}

    def step(self, action):
        if self.fail:
            raise RuntimeError('intentional worker failure')
        self.steps += 1
        done = self.steps >= 2
        info = {'episode_seed': self.episode_seed}
        if done:
            info['episode_record'] = {'result': 'goal', 'seed': self.episode_seed}
        return np.full(19, .25, np.float32), 1., done, False, info

    def close(self):
        if self.close_path is not None:
            Path(self.close_path).write_text('closed', encoding='utf-8')


def make_base(directory):
    directory.mkdir(parents=True)
    (directory/'model.zip').write_bytes(b'fake model, only manifest tests')
    (directory/'config.json').write_text(json.dumps({'role': 'striker', 'history': 1}), encoding='utf-8')
    np.savez(directory/'obs_norm.npz', mean=np.zeros(19), var=np.ones(19), count=5000.,
             clip_obs=10., epsilon=1e-8)


def calibration(base):
    return {'version': CALIBRATION_VERSION, 'initialization': training.INITIALIZATION_VERSION,
            'base_hashes': training.policy_hashes(base), 'seed_base': CALIBRATION_BASE,
            'episode_seeds': [CALIBRATION_BASE*1_000_000+i for i in range(8)],
            'action_head_row_scales': [.1, .5, .8, .02], 'normalization_frozen': True,
            'percentile_uses_only_pre_release_frames': True, 'validation_or_test_used': False}


def metadata():
    return {'hashes': {'test-source': 'hash'}, 'versions': {'python': 'test'}, 'commit': 'parent'}


def record(seed, result='goal', trial=1, fallen=False):
    return {'seed': seed, 'result': result, 'striker_fallen': fallen,
            'reset_info': {'engine_seed': seed % 1000, 'ball_xy': [1., 0.],
                           'start_offsets': {'keeper': [0., 0.], 'striker': [.01, 0.]}, 'trial': trial}}


class RetrainingProtocolTests(unittest.TestCase):
    def test_disjoint_seeds_and_invalid_indices(self):
        namespaces = {split: {training.episode_seed(split, i) for i in (0, 1, 999999)}
                      for split in training.SPLIT_BASES}
        for split, values in namespaces.items():
            for other, peers in namespaces.items():
                if split != other:
                    self.assertFalse(values & peers)
        for split, index in [('unknown', 1), ('train', -1), ('test', True), ('test', 1.5), ('test', 1000000)]:
            with self.subTest(split=split, index=index), self.assertRaises(ValueError):
                training.episode_seed(split, index)

    def test_summary_does_not_treat_misses_as_saves(self):
        summary = training.summarize([record(1), record(2, 'save'), record(3, 'miss'),
                                      record(4, 'timeout'), record(5, 'sim_error')])
        self.assertEqual(summary['episodes'], 5)
        self.assertEqual(summary['goal_rate'], .2)
        self.assertEqual(summary['save_rate'], .2)
        self.assertEqual(summary['sim_errors'], 1)

    def test_paired_goals_all_outcomes_and_session_trial_metadata(self):
        candidate = [record(1), record(2, 'miss'), record(3, 'timeout')]
        baseline = [record(1, 'save', trial=2), record(2, 'goal', trial=2), record(3, 'miss', trial=2)]
        result = training.paired_goals(candidate, baseline)
        self.assertEqual(result['goal_gain'], 0.)
        self.assertEqual(result['candidate_only_goals'], 1)
        self.assertEqual(result['baseline_only_goals'], 1)
        self.assertLessEqual(result['goal_gain_ci95'][0], 0.)
        self.assertGreaterEqual(result['goal_gain_ci95'][1], 0.)

    def test_pairing_rejects_duplicate_unpaired_and_changed_physical_reset(self):
        a = [record(1), record(2)]
        for b in ([record(1)], [record(1), record(1)], [record(1), record(3)]):
            with self.assertRaises(ValueError):
                training.paired_goals(a, b)
        b = deepcopy(a)
        b[1]['reset_info']['ball_xy'] = [.99, 0.]
        with self.assertRaises(ValueError):
            training.paired_goals(a, b)

    def test_selection_prefers_base_on_equal_validation_results(self):
        summary = training.summarize([record(1), record(2, 'save')])
        candidates = [{'update': i, 'summary': deepcopy(summary)} for i in (8, 0, 24, 16)]
        self.assertEqual(training.choose_checkpoint(candidates)['update'], 0)
        candidates[-1]['summary']['goal_rate'] = .75
        self.assertEqual(training.choose_checkpoint(candidates)['update'], 16)
        candidates[-1]['summary']['sim_errors'] = 1
        with self.assertRaises(ValueError):
            training.choose_checkpoint(candidates)

    def test_selection_fall_tie_breaker_precedes_update(self):
        a = {'update': 0, 'summary': {'goal_rate': .5, 'striker_fall_rate': .1, 'sim_errors': 0}}
        b = {'update': 8, 'summary': {'goal_rate': .5, 'striker_fall_rate': 0., 'sim_errors': 0}}
        self.assertEqual(training.choose_checkpoint([a, b])['update'], 8)

    def test_envelope_detects_payload_and_run_tampering(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            path = Path(tmp)/'state.json'
            training._write(path, training._envelope({'update': 8}, 'run'))
            self.assertEqual(training._read_envelope(path, 'run'), {'update': 8})
            with self.assertRaises(ValueError):
                training._read_envelope(path, 'other')
            data = training._read(path)
            data['payload']['update'] = 24
            training._write(path, data)
            with self.assertRaises(ValueError):
                training._read_envelope(path, 'run')

    def test_manifest_freezes_base_profile_calibration_and_resume(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            root = Path(tmp)
            base, output = root/'base', root/'run'
            make_base(base)
            report = root/'calibration.json'
            training._write(report, calibration(base))
            arguments = {'base_policy': base, 'motion_model': training.ROOT/'models/vision/keeper_motion_v1.json',
                         'calibration_report': report, 'workers': 2, 'n_steps': 32, 'updates': 2,
                         'checkpoint_every': 1, 'validation_episodes': 2, 'test_episodes': 3}
            with patch.object(training, 'sources', side_effect=metadata):
                manifest = training.prepare_run(output, **arguments)
                self.assertEqual(manifest['configuration']['reward'], 'striker_visual_reward_v2')
                self.assertFalse(manifest['configuration']['test_may_select_or_fit'])
                self.assertEqual(training.policy_hashes(output/'frozen/striker_v2'), training.policy_hashes(base))
                self.assertEqual(training.prepare_run(output, resume=True, **arguments), manifest)
                with self.assertRaises(ValueError):
                    training.prepare_run(output, **arguments)
                with self.assertRaises(ValueError):
                    training.prepare_run(output, resume=True, **dict(arguments, workers=3))
                (output/'frozen/striker_v2/model.zip').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    training.prepare_run(output, resume=True, **arguments)

    def test_wrong_calibration_source_and_test_tuning_rejected(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            root, base = Path(tmp), Path(tmp)/'base'
            make_base(base)
            for changed in ({'base_hashes': {}}, {'seed_base': 93001},
                            {'validation_or_test_used': True}, {'action_head_row_scales': [1., 1., 1., True]}):
                path = root/'calibration.json'
                training._write(path, dict(calibration(base), **changed))
                with self.subTest(changed=changed), patch.object(training, 'sources', side_effect=metadata):
                    with self.assertRaises(ValueError):
                        training.prepare_run(root/'run', base_policy=base, calibration_report=path)

    def test_wrong_history_and_normalizer_dimensions_rejected(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            base = Path(tmp)/'base'
            make_base(base)
            training._write(base/'config.json', {'role': 'striker', 'history': 6})
            with self.assertRaises(ValueError):
                training.validate_base(base)
            training._write(base/'config.json', {'role': 'striker', 'history': 1})
            np.savez(base/'obs_norm.npz', mean=np.zeros(25), var=np.ones(25), count=1., clip_obs=10., epsilon=1e-8)
            with self.assertRaises(ValueError):
                training.validate_base(base)


class VectorProtocolTests(unittest.TestCase):
    def test_vector_terminal_observation_auto_reset_and_cleanup(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            paths = [str(Path(tmp)/(str(i)+'.closed')) for i in range(2)]
            env = training.make_vector_env({'workers': 2}, factories=[partial(FakePhysicalEnv, i, paths[i])
                                                                     for i in range(2)], timeout_s=30.)
            processes = list(env.processes)
            try:
                self.assertEqual(env.reset().shape, (2, 19))
                _, rewards, done, _ = env.step(np.zeros((2, 4), np.float32))
                np.testing.assert_equal(done, [False, False])
                observation, rewards, done, infos = env.step(np.zeros((2, 4), np.float32))
                np.testing.assert_equal(done, [True, True])
                np.testing.assert_equal(observation, np.zeros((2, 19)))
                np.testing.assert_equal(infos[0]['terminal_observation'], np.full(19, .25, np.float32))
                self.assertEqual(infos[0]['episode'], {'r': 2., 'l': 2})
                self.assertFalse(infos[0]['TimeLimit.truncated'])
                self.assertNotEqual(env.reset_infos[0]['episode_seed'], infos[0]['episode_seed'])
            finally:
                env.close()
            self.assertTrue(all(not p.is_alive() for p in processes))
            self.assertTrue(all(Path(path).read_text() == 'closed' for path in paths))

    def test_worker_exception_is_reported_and_every_process_closed(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            path = str(Path(tmp)/'closed')
            env = training.make_vector_env({'workers': 1}, factories=[partial(FakePhysicalEnv, 0, path, True)], timeout_s=30.)
            processes = list(env.processes)
            try:
                env.reset()
                with self.assertRaisesRegex(RuntimeError, 'intentional worker failure'):
                    env.step(np.zeros((1, 4), np.float32))
            finally:
                env.close()
            self.assertTrue(all(not p.is_alive() for p in processes))
            self.assertTrue(Path(path).is_file())

    def test_rng_snapshot_restore_preserves_all_three_generators(self):
        import random
        import torch
        state = training._rng_snapshot()
        first = (random.random(), float(np.random.random()), float(torch.rand(1)))
        training._rng_restore(state)
        second = (random.random(), float(np.random.random()), float(torch.rand(1)))
        self.assertEqual(first, second)

    def test_warm_load_matches_v2_normalization_and_retains_optimizer(self):
        configuration = {'workers': 1, 'ppo': {'n_steps': 32, 'batch_size': 32,
                                              'learning_rate': 1e-4, 'clip_range': .15,
                                              'n_epochs': 10, 'target_kl': .03,
                                              'ent_coef': 0., 'gamma': .99, 'gae_lambda': .95}}
        vector = training.make_vector_env(configuration, factories=[FakePhysicalEnv], timeout_s=30.)
        normalizer = None
        try:
            directory = training.ROOT/'models/rl/striker_v2'
            model, normalizer = training.load_warm_start(directory, vector, configuration)
            self.assertFalse(normalizer.training)
            self.assertFalse(normalizer.norm_reward)
            self.assertEqual(model.n_steps, 32)
            self.assertGreater(len(model.policy.optimizer.state), 0)
            with np.load(directory/'obs_norm.npz') as stats:
                observation = np.zeros((1, 19), np.float32)
                expected = np.clip((observation-stats['mean'])/np.sqrt(stats['var']+stats['epsilon']), -10., 10.)
                np.testing.assert_allclose(normalizer.normalize_obs(observation), expected, rtol=1e-6)
                np.testing.assert_equal(normalizer.obs_rms.mean, stats['mean'])
            low, high = .15, .45
            with __import__('torch').no_grad():
                model.policy.log_std[0] = -100.
                model.policy.log_std[1] = 100.
            training.clamp_std(model, {'action_std_bounds': [low, high]})
            values = model.policy.log_std.detach().exp().numpy()
            self.assertTrue((values >= low-1e-7).all())
            self.assertTrue((values <= high+1e-7).all())
        finally:
            if normalizer is not None:
                normalizer.close()
            else:
                vector.close()


if __name__ == '__main__':
    unittest.main()
