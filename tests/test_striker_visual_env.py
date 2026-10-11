"""Camera opponent routing, transfer contracts and failure cleanup without physics."""
import hashlib
from types import SimpleNamespace
import unittest

import numpy as np

from penalty.keeper_defenses import DEFENSE_CATALOG_VERSION, DEFENSE_NAMES
from penalty.motion_contract import ROOT
from penalty.rl.contracts import RLObservation, StrikerAction, KeeperAction, clip_observation
from penalty.rl.striker_visual_env import (MAX_EPISODE_STEPS, RESULTS,
    StrikerVisualKeeperEnv, scheduled_episode_seed, striker_visual_reward_v2)
from penalty.rl.wrappers import PerceptionNoise


def observation(time=0.):
    return RLObservation(time, .2, 0., 0., 0., 0., 2., .1, 0.,
                         0., .2, -.2, 0., 0., 0., 0., 0., 28.-time)


def vision_state(time=0., crossing=-.1):
    return {'version': 'vision_state_v2', 'time_s': time, 'capture_s': time,
            'remaining_time': 28.-time,
            'ball': {'visible': True, 'position_valid': True, 'position': [1., crossing],
                     'velocity_valid': True, 'velocity': [-1., 0.],
                     'velocity_covariance': [[0., 0.], [0., 0.]], 'age_s': 0.},
            'sensor_provenance': 'own simulated sensors test',
            'inference_clock': 'simulated physics pauses during inference'}


class FakeProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


class FakeWorld:
    control_dt = .1

    def __init__(self, **options):
        self.options = options
        self.sim = SimpleNamespace(model=object(), fallen={'striker': False}, bridges={})
        self.session = SimpleNamespace(sim=self.sim)
        self.calls, self.reset_random, self.closed = [], [], False
        self.results = [None, 'save']
        self.raise_step = None
        self.change_model_on_reset = False
        self._new_bridges()

    def _new_bridges(self):
        self.sim.bridges = {role: SimpleNamespace(process=FakeProcess(), hello={
            'capabilities': [DEFENSE_CATALOG_VERSION], 'keeper_defenses': list(DEFENSE_NAMES)})
            for role in ('striker', 'keeper')}

    def reset(self, rng, options):
        self.closed = False
        if self.session is None:
            self._new_bridges()
            self.session = SimpleNamespace(sim=self.sim)
        if self.change_model_on_reset:
            self.sim.model = object()
        self.calls = []
        self.time = 0.
        self.reset_random.append(rng.random(8).tolist())
        self.commit_defense = None
        return observation(), observation(), {'engine_seed': 23, 'ball_xy': [1., 0.]}

    def duel_step(self, striker_action, keeper_action):
        if self.raise_step:
            raise self.raise_step
        self.calls.append((striker_action, keeper_action))
        if keeper_action.fires:
            self.commit_defense = getattr(keeper_action, 'defense', 'center')
        result = self.results[min(len(self.calls)-1, len(self.results)-1)]
        self.time += .1
        return observation(self.time), observation(self.time), result, {}

    def record(self):
        return {'commit_defense': self.commit_defense}

    def close(self):
        self.closed = True
        if self.session is not None:
            for bridge in self.sim.bridges.values():
                bridge.process.returncode = 0
        self.session = None


class FakeCamera:
    def __init__(self, world):
        self.world, self.model = world, world.sim.model
        self.detector = SimpleNamespace(process=FakeProcess(), hello={'model_sha256': 'test'})
        self.closed, self.reset_count, self.observe_count = False, 0, 0
        self.failure = None
        self.scene_secret = {'opponent_ideal': [1000., 999.]}

    def reset(self):
        self.reset_count += 1

    def observe(self):
        self.observe_count += 1
        if self.failure:
            raise self.failure
        return vision_state(self.world.time) | self.scene_secret

    def close(self):
        self.closed = True
        self.detector.process.returncode = 0


class StrikerVisualTests(unittest.TestCase):
    def env(self, **options):
        env = StrikerVisualKeeperEnv(world_factory=FakeWorld, camera_factory=FakeCamera,
                                    **options)
        self.addCleanup(env.close)
        return env

    def test_compatible_shapes_and_explicit_new_goal_reward(self):
        env = self.env()
        obs, info = env.reset(seed=91)
        self.assertEqual((obs.shape, env.action_space.shape), ((19,), (4,)))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(info['observation'], 'rl_obs_v1')
        self.assertEqual(info['striker_action'], 'striker_action_v1')
        self.assertEqual(info['reward'], 'striker_visual_reward_v2')
        self.assertEqual(info['keeper_policy']['core_policy'], 'vision_keeper_v2')
        self.assertEqual(info['keeper_policy']['mode'], 'balanced')
        self.assertIsNone(info['keeper_motion_profile'])
        self.assertFalse(info['hardware_validated'])
        self.assertIn('privileged scene', info['perception']['striker'])
        self.assertIn('rendered RGB', info['perception']['keeper'])

    def test_camera_decides_before_physics_and_balanced_mirrors_negative_center(self):
        env = self.env(perception_noise=False)
        env.reset(seed=11)
        action = StrikerAction(.4, 1, .7, 1.).to_normalized()
        _, reward, terminated, truncated, info = env.step(action)
        self.assertEqual((reward, terminated, truncated), (-.02, False, False))
        self.assertFalse(env.world.calls[0][1].fires)
        _, reward, terminated, truncated, info = env.step(action)
        self.assertEqual((reward, terminated, truncated), (-10.02, True, False))
        self.assertEqual(env.world.calls[1][0], StrikerAction.from_normalized(action))
        self.assertEqual(env.world.calls[1][1].defense, 'center_right')
        self.assertTrue(env.world.calls[1][1].fires)
        record = info['episode_record']
        self.assertEqual(record['result'], 'save')
        self.assertEqual(record['outcome'], 'save')
        self.assertEqual(record['first_trigger']['time_s'], .1)
        self.assertEqual(record['seed'], 11)
        self.assertEqual(record['visible_frames'], 2)
        self.assertEqual(record['runtime_provenance']['sensor'], 'own simulated sensors test')
        self.assertEqual(env.camera.observe_count, 2)
        with self.assertRaises(RuntimeError):
            env.step(action)

    def test_striker_noise_matches_camera_evaluation_rng_without_world_draw_coupling(self):
        env = self.env()
        first, _ = env.reset(seed=456)
        noise, rng = PerceptionNoise(), np.random.default_rng([456, 0])
        expected = noise(clip_observation(observation().to_array()), rng)
        np.testing.assert_array_equal(first, expected)
        next_obs, *_ = env.step(np.array([0., -1., .4, 1.]))
        np.testing.assert_array_equal(next_obs,
                                      noise(clip_observation(observation(.1).to_array()), rng))
        repeated, _ = env.reset(seed=456)
        np.testing.assert_array_equal(repeated, first)
        self.assertEqual(env.world.reset_random[-2], env.world.reset_random[-1])

    def test_unseeded_worker_schedule_disjoint_from_eval_and_other_workers(self):
        workers = [self.env(worker_index=i, worker_count=3, episode_start=9) for i in range(3)]
        values = []
        for env in workers:
            env.reset()
            values.append(env.episode_seed)
            env.reset(seed=9_900_000)
            env.reset()
            values.append(env.episode_seed)
        self.assertEqual(sorted(values), list(range(91_001_000_009, 91_001_000_015)))
        self.assertNotEqual(scheduled_episode_seed('train', 0), scheduled_episode_seed('test', 0))
        for split, index, base in (('unknown', 0, None), ('test', -1, None),
                                    ('test', 1_000_000, None), ('train', True, None),
                                    ('train', 0, 0), ('train', 0, True)):
            with self.assertRaises(ValueError):
                scheduled_episode_seed(split, index, base)

    def test_reward_has_no_wait_or_timeout_advantage_over_another_non_goal(self):
        self.assertEqual(striker_visual_reward_v2(None), -.02)
        self.assertEqual(striker_visual_reward_v2('goal'), 9.98)
        for result in RESULTS:
            if result != 'goal':
                self.assertEqual(striker_visual_reward_v2(result), -10.02)
        with self.assertRaises(ValueError):
            striker_visual_reward_v2('unknown')

    def test_true_goal_and_all_failures_are_finite_horizon_terminal(self):
        for result in RESULTS:
            with self.subTest(result=result):
                env = self.env()
                env.world.results = [result]
                env.reset(seed=7)
                _, reward, terminal, truncated, info = env.step(np.zeros(4))
                self.assertTrue(terminal)
                self.assertFalse(truncated)
                self.assertEqual(info['result'], result)
                self.assertEqual(info['episode_record']['seed'], 7)
                self.assertEqual(reward, 9.98 if result == 'goal' else -10.02)

    def test_camera_or_physics_failure_collects_all_children(self):
        for failure_source in ('camera', 'physics'):
            with self.subTest(source=failure_source):
                env = self.env()
                env.reset(seed=9)
                camera = env.camera
                children = env._processes()
                if failure_source == 'camera':
                    camera.failure = KeyError('invalid sensor frame')
                else:
                    env.world.raise_step = RuntimeError('native sensor limit')
                _, reward, terminal, truncated, info = env.step(np.zeros(4))
                self.assertEqual((reward, terminal, truncated), (-10.02, True, False))
                self.assertEqual(info['result'], 'sim_error')
                self.assertTrue(info['children_collected'])
                self.assertTrue(all(child.poll() is not None for child in children))
                self.assertTrue(camera.closed)
                self.assertTrue(env.world.closed)

    def test_capability_failure_on_reset_closes_world(self):
        env = self.env()
        env.world.sim.bridges['keeper'].hello['capabilities'] = []
        with self.assertRaisesRegex(RuntimeError, 'capability'):
            env.reset(seed=1)
        self.assertTrue(env.world.closed)

    def test_reset_reuses_camera_only_for_identical_model_and_cleans_replacements(self):
        env = self.env()
        env.reset(seed=1)
        camera = env.camera
        env.reset(seed=2)
        self.assertIs(env.camera, camera)
        self.assertEqual(camera.reset_count, 2)
        env.world.change_model_on_reset = True
        env.reset(seed=3)
        self.assertTrue(camera.closed)
        self.assertIsNot(env.camera, camera)

    def test_scripted_fsm_is_declared_scene_based_and_runs_without_camera(self):
        env = self.env(keeper_mode='scripted')
        _, info = env.reset(seed=11)
        self.assertIsNone(env.camera)
        self.assertFalse(info['keeper_policy']['official_bhuman_behavior'])
        self.assertIsNone(info['camera_pipeline'])
        self.assertIn('privileged scene', info['perception']['keeper'])
        env.step(np.zeros(4))
        self.assertIsInstance(env.world.calls[0][1], KeeperAction)

    def test_bounded_decision_guard_fails_closed_and_invalid_action_has_no_motor_effect(self):
        env = self.env()
        env.reset(seed=1)
        with self.assertRaises(ValueError):
            env.step(np.array([np.nan, 0., 0., 0.]))
        self.assertEqual(env.world.calls, [])
        env._steps = MAX_EPISODE_STEPS
        *_, info = env.step(np.zeros(4))
        self.assertEqual(info['result'], 'sim_error')
        self.assertTrue(info['children_collected'])

    def test_configuration_rejects_inconsistent_motion_and_worker_options(self):
        for options in ({'keeper_mode': 'unknown'}, {'motion_model': 'profile.json'},
                        {'worker_count': 0}, {'worker_index': 2, 'worker_count': 2},
                        {'perception_noise': 1}, {'world_options': {'keeper_gate': True}}):
            with self.assertRaises(ValueError):
                self.env(**options)

    def test_motion_evaluation_loads_exact_frozen_profile_without_changing_balanced_prediction(self):
        path = ROOT/'models/vision/keeper_motion_v1.json'
        original = path.read_bytes()
        env = self.env(keeper_mode='motion', motion_model=path)
        _, info = env.reset(seed=5)
        profile = info['keeper_motion_profile']
        self.assertEqual(profile['sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(profile['parameters']['jump_time_scale'], .85)
        self.assertEqual(env.world.options['motion_parameters'].jump_time_scale, .85)
        self.assertEqual(info['keeper_policy']['mode'], 'balanced')
        self.assertEqual(info['keeper_policy']['core_policy'], 'vision_keeper_v2')
        self.assertEqual(path.read_bytes(), original)

    def test_terminal_provenance_failure_also_collects_children(self):
        env = self.env()
        env.world.results = ['save']
        env.reset(seed=4)
        children = env._processes()

        def broken_record():
            raise RuntimeError('native profile confirmation missing')
        env.world.record = broken_record
        with self.assertRaisesRegex(RuntimeError, 'confirmation'):
            env.step(np.zeros(4))
        self.assertTrue(all(child.poll() is not None for child in children))
        self.assertTrue(env.world.closed)


if __name__ == '__main__':
    unittest.main()
