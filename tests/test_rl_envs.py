"""Gymnasium API, seeding, wrappers and baselines on the synthetic backend."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

from gymnasium.utils.env_checker import check_env
import numpy as np

from penalty.motion_contract import ROOT
from penalty.rl.contracts import FIELD, OBSERVATION_FIELDS
from penalty.rl.evaluation import evaluate, make_env, rollout
from penalty.rl.keeper_env import PenaltyKeeperEnv
from penalty.rl.policies.baselines import make_policy
from penalty.rl.run_metadata import run_metadata
from penalty.rl.striker_env import PenaltyStrikerEnv
from penalty.rl.wrappers import DelayedObservationWrapper, ObservationHistory, PerceptionNoiseWrapper


class SyntheticEnvTests(unittest.TestCase):
    def test_gymnasium_api(self):
        for opponent in ('empty', 'static', 'random', 'scripted'):
            with self.subTest(opponent=opponent):
                check_env(PenaltyStrikerEnv(opponent=opponent), skip_render_check=True)
        check_env(PenaltyKeeperEnv(), skip_render_check=True)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            PenaltyStrikerEnv(opponent='nobody')
        with self.assertRaises(ValueError):
            PenaltyStrikerEnv(backend='unity')
        with self.assertRaises(ValueError):
            PenaltyKeeperEnv(reward_id='keeper_reward_v0')

    def test_seeded_episodes_are_reproducible(self):
        for env_type in (PenaltyStrikerEnv, PenaltyKeeperEnv):
            env = env_type()
            policy = make_policy('random', env.role, seed=4)
            first = [rollout(env, policy, seed) for seed in range(20)]
            policy = make_policy('random', env.role, seed=4)
            second = [rollout(env, policy, seed) for seed in range(20)]
            self.assertEqual(first, second)
            self.assertGreater(len({r['outcome'] for r in first}), 1)

    def test_info_reports_versions_and_terminal_record(self):
        env = PenaltyStrikerEnv(opponent='empty')
        observation, info = env.reset(seed=0)
        self.assertEqual(observation.shape, (len(OBSERVATION_FIELDS),))
        self.assertEqual(info['backend'], 'synthetic')
        self.assertEqual(info['reward'], 'striker_reward_v1')
        self.assertEqual(info['observation'], 'rl_obs_v1')
        policy = make_policy('scripted', 'striker')
        _, reward, terminated, truncated, info = env.step(policy(observation))
        self.assertTrue(terminated and not truncated)
        self.assertIn(info['outcome'], ('goal', 'fall'))
        self.assertEqual(info['episode_record']['steps'], 1)
        with self.assertRaises(RuntimeError):
            env.step(policy(observation))

    def test_striker_safety_requires_alignment(self):
        env = PenaltyStrikerEnv(opponent='empty')
        env.reset(seed=1)
        aimed = np.array([1., -1., 0., 1.], dtype=np.float32)  # far left, fire
        _, _, terminated, _, info = env.step(aimed)
        self.assertFalse(terminated)
        self.assertEqual(info['safety'], 'not_aligned')

    def test_keeper_baselines(self):
        geometric = evaluate('keeper', 'geometric', episodes=300, seeds=(0,))['summary']['rates']
        scripted = evaluate('keeper', 'scripted', episodes=300, seeds=(0,))['summary']['rates']
        random = evaluate('keeper', 'random', episodes=300, seeds=(0,))['summary']['rates']
        self.assertGreater(geometric['save_rate'], scripted['save_rate'])
        self.assertGreater(scripted['save_rate'], random['save_rate'])
        self.assertLess(geometric['wrong_side_rate'], .1)

    def test_evaluate_with_workers_matches_serial(self):
        serial = evaluate('striker', 'random', episodes=12, seeds=(0, 1), workers=1)
        parallel = evaluate('striker', 'random', episodes=12, seeds=(0, 1), workers=2)
        self.assertEqual(serial['summary']['counts'], parallel['summary']['counts'])
        self.assertEqual(serial['throughput']['episodes'], 24)


class WrapperTests(unittest.TestCase):
    def test_history_stacks_oldest_first(self):
        env = ObservationHistory(PenaltyKeeperEnv(), length=6)
        size = len(OBSERVATION_FIELDS)
        observation, _ = env.reset(seed=0)
        self.assertEqual(observation.shape, (6*size,))
        self.assertTrue(env.observation_space.contains(observation))
        wait = np.array([0., -1., 0.], dtype=np.float32)
        newest, *_ = env.step(wait)
        np.testing.assert_array_equal(newest[:5*size], observation[size:])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            check_env(env, skip_render_check=True)

    def test_delay(self):
        env = DelayedObservationWrapper(PenaltyKeeperEnv(), delay_steps=2)
        first, _ = env.reset(seed=0)
        wait = np.array([0., -1., 0.], dtype=np.float32)
        a, *_ = env.step(wait)
        b, *_ = env.step(wait)
        c, *_ = env.step(wait)
        time = FIELD['time_s']
        self.assertEqual((first[time], a[time], b[time]), (0., 0., 0.))
        self.assertAlmostEqual(float(c[time]), .1, places=5)

    def test_noise_and_dropout_are_seeded(self):
        def run():
            env = PerceptionNoiseWrapper(PenaltyKeeperEnv(), dropout=.5)
            values, _ = env.reset(seed=7)
            frames = [values]
            for _ in range(10):
                values, _, terminated, truncated, _ = env.step(np.array([0., -1., 0.]))
                frames.append(values)
                if terminated or truncated:
                    break
            return np.array(frames)
        first, second = run(), run()
        np.testing.assert_array_equal(first, second)
        visible = first[:, FIELD['ball_visible']]
        self.assertIn(0., visible)
        lost = int(np.flatnonzero(visible == 0.)[0])
        self.assertGreater(first[lost, FIELD['ball_age']], 0.)
        self.assertEqual(first[lost, FIELD['ball_x']], first[lost-1, FIELD['ball_x']])


class ToolTests(unittest.TestCase):
    def test_metadata(self):
        metadata = run_metadata(seed=3, model=None)
        self.assertEqual(metadata['seed'], 3)
        self.assertEqual(metadata['versions']['observation'], 'rl_obs_v1')
        self.assertIn('penalty/rl/rewards.py', metadata['hashes'])

    def test_evaluate_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'report.json'
            subprocess.run([sys.executable, str(ROOT/'tools'/'evaluate_rl.py'), '--role', 'keeper',
                            '--policy', 'geometric', '--episodes', '20', '--seeds', '0', '1',
                            '--output', str(output)], check=True, capture_output=True, cwd=ROOT,
                           env=dict(os.environ, PYTHONPATH=str(ROOT)), timeout=120)
            report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(report['summary']['episodes'], 40)
        self.assertEqual(set(report['per_seed']), {'0', '1'})
        self.assertEqual(report['metadata']['policy'], 'geometric')


if __name__ == '__main__':
    unittest.main()
