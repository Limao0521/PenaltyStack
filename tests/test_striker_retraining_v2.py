"""Preserving PPO protocol, failure evidence and retention, without physics."""
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

from penalty.rl import striker_retraining_v2 as training
from penalty.rl.contracts import OBS_LOW, OBS_HIGH


class FakePhysicalEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, index=0, close_path=None, fail=False, error_result=False):
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
        self.worker_index, self.ordinal, self.steps = index, 0, 0
        self.close_path, self.fail, self.error_result = close_path, fail, error_result

    def reset(self, *, seed=None, options=None):
        self.ordinal += 1
        self.steps = 0
        self.episode_seed = training.episode_seed('train', self.ordinal*2+self.worker_index)
        return np.zeros(19, np.float32), {'episode_seed': self.episode_seed}

    def step(self, action):
        if self.fail:
            import sys
            print('worker stdout detail')
            print('worker stderr detail', file=sys.stderr)
            raise RuntimeError('intentional worker failure')
        self.steps += 1
        done = self.steps >= 2 or self.error_result
        info = {'episode_seed': self.episode_seed}
        if done:
            result = 'sim_error' if self.error_result else 'goal'
            info.update(result=result, episode_record={'result': result, 'seed': self.episode_seed})
            if self.error_result:
                info.update(sim_error='intentional physical error', sim_error_traceback='original physics trace')
        return np.full(19, .25, np.float32), 0. if self.error_result else 1., done, False, info

    def close(self):
        if self.close_path is not None:
            Path(self.close_path).write_text('closed', encoding='utf-8')


def make_base(directory):
    directory.mkdir(parents=True)
    (directory/'model.zip').write_bytes(b'fake model, only manifest tests')
    (directory/'config.json').write_text(json.dumps({'role': 'striker', 'history': 1}), encoding='utf-8')
    np.savez(directory/'obs_norm.npz', mean=np.zeros(19), var=np.ones(19), count=5000., clip_obs=10., epsilon=1e-8)


def metadata():
    return {'hashes': {'test-source': 'hash'}, 'versions': {'python': 'test'}, 'commit': 'parent'}


def candidate(update, primary, retention):
    return {'update': update, 'summary': {'goal_rate': primary, 'striker_fall_rate': 0., 'sim_errors': 0},
            'retention_summary': {'goal_rate': retention, 'striker_fall_rate': 0., 'sim_errors': 0}}


class RetrainingProtocolTests(unittest.TestCase):
    def test_new_seed_namespaces_are_disjoint_from_prior_experiment(self):
        from penalty.rl.striker_retraining import SPLIT_BASES as previous
        self.assertFalse(set(previous.values()) & set(training.SPLIT_BASES.values()))
        self.assertEqual(training.episode_seed('test', 99), 103001000099)
        for split, index in [('unknown', 1), ('train', -1), ('test', True), ('test', 1000000)]:
            with self.subTest(split=split, index=index), self.assertRaises(ValueError):
                training.episode_seed(split, index)

    def test_retention_gate_prefers_eligible_over_larger_primary_gain(self):
        base = candidate(0, .2, .9)
        good, degraded = candidate(32, .3, .8), candidate(64, .8, .79)
        selected, promotion, passed = training.choose_retained_checkpoint([degraded, good], base)
        self.assertIs(selected, good)
        self.assertIs(promotion, good)
        self.assertTrue(passed)

    def test_all_retention_failures_only_allow_diagnostic_candidate(self):
        base = candidate(0, .2, .9)
        a, b = candidate(32, .3, .7), candidate(64, .4, .6)
        selected, promotion, passed = training.choose_retained_checkpoint([a, b], base)
        self.assertIs(selected, b)
        self.assertIs(promotion, base)
        self.assertFalse(passed)

    def test_retention_cannot_promote_lower_primary_and_rejects_errors(self):
        base, trained = candidate(0, .2, .9), candidate(32, .1, .95)
        _, promotion, passed = training.choose_retained_checkpoint([trained], base)
        self.assertIs(promotion, base)
        self.assertTrue(passed)
        trained['retention_summary']['sim_errors'] = 1
        with self.assertRaises(ValueError):
            training.choose_retained_checkpoint([trained], base)

    def test_manifest_freezes_actor_new_objective_mixture_and_resume(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            root, base = Path(tmp), Path(tmp)/'base'
            make_base(base)
            arguments = {'base_policy': base, 'workers': 6, 'n_steps': 32, 'updates': 2,
                         'checkpoint_every': 1, 'validation_episodes': 2, 'test_episodes': 3}
            with patch.object(training, 'sources', side_effect=metadata):
                manifest = training.prepare_run(root/'run', **arguments)
                config = manifest['configuration']
                self.assertEqual(config['reward'], 'striker_visual_reward_v3')
                self.assertEqual(config['ppo']['gamma'], 1.)
                self.assertEqual(config['training_opponents_by_worker'], ['balanced']*5+['scripted'])
                self.assertTrue(config['actor_weights_preserved_initially'])
                self.assertFalse(config['head_rows_rescaled'])
                self.assertFalse(config['async_training_compatible'])
                self.assertFalse(config['test_may_select_or_fit'])
                self.assertEqual(training.policy_hashes(root/'run/frozen/striker_v2'), training.policy_hashes(base))
                self.assertEqual(training.prepare_run(root/'run', resume=True, **arguments), manifest)
                with self.assertRaises(ValueError):
                    training.prepare_run(root/'run', resume=True, **dict(arguments, retention_max_drop=.2))
                (root/'run/frozen/striker_v2/model.zip').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    training.prepare_run(root/'run', resume=True, **arguments)

    def test_reservation_covers_every_possible_one_step_terminal(self):
        config = {'workers': 6, 'ppo': {'n_steps': 128}}
        state = {'completed_updates': 4, 'episode_start': 70, 'checkpoint': 'cp4',
                 'rng': {'saved': True}, 'training_transitions': 3072}
        reservation = training.rollout_reservation(state, config)
        self.assertEqual((reservation['reserved_indices'], reservation['resume_episode_start']), (774, 844))
        for worker in range(6):
            for ordinal in range(129):
                self.assertLess(70+worker+ordinal*6, reservation['resume_episode_start'])
        recovered = training.recover_incomplete_rollout(dict(state, inflight_rollout=reservation))
        self.assertEqual(recovered['episode_start'], 844)
        for key in ('checkpoint', 'rng', 'training_transitions', 'completed_updates'):
            self.assertEqual(recovered[key], state[key])
        self.assertNotIn('inflight_rollout', recovered)
        self.assertEqual(state['episode_start'], 70)

    def test_checkpoint_envelopes_reject_tampered_rng(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            path = Path(tmp)/'state.json'
            training._write(path, training._envelope({'rng': {'seed': 1}}, 'run'))
            value = training._read(path)
            value['payload']['rng']['seed'] = 2
            training._write(path, value)
            with self.assertRaises(ValueError):
                training._read_envelope(path, 'run')


class VectorAndFailureTests(unittest.TestCase):
    def test_terminal_auto_reset_and_cleanup(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            paths = [str(Path(tmp)/(str(i)+'.closed')) for i in range(2)]
            env = training.make_vector_env({'workers': 2}, factories=[partial(FakePhysicalEnv, i, paths[i])
                                                                     for i in range(2)], timeout_s=30.)
            processes = list(env.processes)
            try:
                env.reset()
                env.step(np.zeros((2, 4), np.float32))
                observation, _, done, infos = env.step(np.zeros((2, 4), np.float32))
                np.testing.assert_equal(done, [True, True])
                np.testing.assert_equal(observation, np.zeros((2, 19)))
                np.testing.assert_equal(infos[0]['terminal_observation'], np.full(19, .25, np.float32))
                self.assertFalse(infos[0]['TimeLimit.truncated'])
            finally:
                env.close()
            self.assertTrue(all(not p.is_alive() for p in processes))
            self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_worker_exception_preserves_seed_stdout_stderr_and_trace(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            path = str(Path(tmp)/'closed')
            env = training.make_vector_env({'workers': 1}, factories=[partial(FakePhysicalEnv, 0, path, True)],
                                           timeout_s=30., failure_output=tmp)
            processes = list(env.processes)
            try:
                env.reset()
                with self.assertRaisesRegex(training.PhysicalWorkerFailure, 'intentional worker failure'):
                    env.step(np.zeros((1, 4), np.float32))
            finally:
                env.close()
            saved = json.loads((Path(tmp)/'failures.jsonl').read_text())
            self.assertEqual(saved['seed'], training.episode_seed('train', 2))
            self.assertIn('worker stdout detail', saved['stdout'])
            self.assertIn('worker stderr detail', saved['stderr'])
            self.assertIn('raise RuntimeError', saved['traceback'])
            self.assertFalse(saved['error_transition_returned_to_ppo'])
            self.assertTrue(all(not p.is_alive() for p in processes))
            self.assertTrue(Path(path).is_file())

    def test_sim_error_is_persisted_without_returning_transition_or_autoreset(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            env = training.make_vector_env({'workers': 1}, factories=[partial(FakePhysicalEnv, error_result=True)],
                                           timeout_s=30., failure_output=tmp)
            try:
                env.reset()
                with self.assertRaises(training.PhysicalWorkerFailure):
                    env.step(np.zeros((1, 4), np.float32))
            finally:
                env.close()
            saved = json.loads((Path(tmp)/'failures.jsonl').read_text())
            self.assertEqual(saved['episode_record']['result'], 'sim_error')
            self.assertEqual(saved['seed'], training.episode_seed('train', 2))
            self.assertEqual(saved['last_info']['sim_error_traceback'], 'original physics trace')
            self.assertFalse(saved['error_transition_returned_to_ppo'])

    def test_failed_rollout_never_updates_optimizer_and_requires_explicit_resume(self):
        original_vector = training.make_vector_env
        def fake_vector(configuration, episode_start=0, **kwargs):
            return original_vector(configuration, episode_start,
                                   factories=[partial(FakePhysicalEnv, error_result=True)],
                                   timeout_s=30., **kwargs)
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            output = Path(tmp)/'run'
            with patch.object(training, 'sources', side_effect=metadata), \
                    patch.object(training, 'make_vector_env', side_effect=fake_vector), \
                    patch('penalty.rl.training._prevent_sleep'):
                manifest = training.prepare_run(output, workers=1, n_steps=32, updates=1,
                                                checkpoint_every=1, validation_episodes=1, test_episodes=1)
                with self.assertRaises(training.PhysicalWorkerFailure):
                    training.train(output, manifest, progress=lambda value: None)
                state = training._read_envelope(output/'progress.json', manifest['fingerprint'])
                self.assertEqual((state['completed_updates'], state['training_transitions']), (0, 0))
                self.assertEqual(state['episode_start'], 33)
                self.assertFalse(list((output/'checkpoints').glob('update_*')))
                from stable_baselines3 import PPO
                initialized = PPO.load(output/'initialized/model.zip', device='cpu')
                self.assertEqual(len(initialized.policy.optimizer.state), 0)
                with self.assertRaisesRegex(RuntimeError, 'explicitly use --resume'):
                    training.train(output, manifest)

    def test_initialization_uses_one_frozen_normalization_and_fresh_optimizer(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        vector = DummyVecEnv([FakePhysicalEnv])
        normalizer = None
        configuration = {'ppo': {'n_steps': 32, 'batch_size': 32, 'learning_rate': 3e-5,
                                  'clip_range': .05, 'n_epochs': 4, 'target_kl': .01,
                                  'ent_coef': 0., 'gamma': 1., 'gae_lambda': .95}}
        try:
            directory = training.ROOT/'models/rl/striker_v2'
            model, normalizer = training.load_warm_start(directory, vector, configuration, initialize=True)
            self.assertFalse(normalizer.training)
            self.assertFalse(normalizer.norm_reward)
            self.assertEqual(len(model.policy.optimizer.state), 0)
            proof = model._preserving_initialization_proof
            self.assertTrue(proof['actor_parameters_byte_identical'])
            self.assertEqual(proof['base_actor_sha256'], proof['initialized_actor_sha256'])
            with np.load(directory/'obs_norm.npz') as stats:
                observation = np.zeros((1, 19), np.float32)
                expected = np.clip((observation-stats['mean'])/np.sqrt(stats['var']+stats['epsilon']), -10., 10.)
                np.testing.assert_allclose(normalizer.normalize_obs(observation), expected, rtol=1e-6)
                np.testing.assert_equal(normalizer.obs_rms.mean, stats['mean'])
        finally:
            (normalizer or vector).close()

    def test_checkpoint_save_failure_retains_previous_published_state(self):
        original_vector, original_save = training.make_vector_env, training._save_policy
        def fake_vector(configuration, episode_start=0, **kwargs):
            return original_vector(configuration, episode_start, factories=[FakePhysicalEnv],
                                   timeout_s=30., **kwargs)
        def fail_update_save(model, normalizer, directory, update, manifest, resume_state=None):
            if update:
                raise OSError('intentional checkpoint save failure')
            return original_save(model, normalizer, directory, update, manifest, resume_state)
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            output = Path(tmp)/'run'
            with patch.object(training, 'sources', side_effect=metadata), \
                    patch.object(training, 'make_vector_env', side_effect=fake_vector), \
                    patch.object(training, '_save_policy', side_effect=fail_update_save), \
                    patch('penalty.rl.training._prevent_sleep'):
                manifest = training.prepare_run(output, workers=1, n_steps=32, updates=1,
                                                checkpoint_every=1, validation_episodes=1, test_episodes=1)
                with self.assertRaisesRegex(OSError, 'checkpoint save failure'):
                    training.train(output, manifest, progress=lambda value: None)
                state = training._read_envelope(output/'progress.json', manifest['fingerprint'])
                self.assertEqual((state['completed_updates'], state['training_transitions']), (0, 0))
                self.assertEqual(state['checkpoint'], 'initialized')
                self.assertEqual(state['checkpoint_sha256'], training.policy_hashes(output/'initialized'))
                self.assertEqual(state['episode_start'], 33)
                failure = json.loads((output/'failures.jsonl').read_text())
                self.assertEqual(failure['update_phase'], 'saving_checkpoint')
                self.assertTrue(failure['optimizer_update_may_have_started'])

    def test_failure_append_handles_numpy_and_retains_all_rows(self):
        with tempfile.TemporaryDirectory(dir=training.ROOT) as tmp:
            training.persist_failure(tmp, {'observation': np.array([1., 2.]), 'value': np.float32(1.)})
            training.persist_failure(tmp, {'message': 'second error'})
            rows = [json.loads(row) for row in (Path(tmp)/'failures.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['observation'], [1., 2.])
            self.assertEqual(rows[1]['message'], 'second error')


if __name__ == '__main__':
    unittest.main()
