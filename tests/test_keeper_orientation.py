import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from penalty.keeper_pose_vision import VisionPoseKeeper
from penalty.opponent_orientation import VisualYawEstimator, DEFAULT_MODEL, body_feature, wrap
from penalty.rl.orientation_contracts import OrientationObservation, from_vision_state
from penalty.sim_sensors import OwnSensorReader, ENCODERS
from tests.test_keeper_vision import state

ROOT = Path(__file__).resolve().parents[1]


def pose_state(t, y=.285, vx=0., yaw_valid=True):
    sample = state(t, y=y, vx=vx)
    sample['version'] = 'vision_state_v2'
    sample['opponent'] = {'position': (1.2, y), 'position_valid': True,
        'position_covariance': ((.005, 0.), (0., .005)), 'yaw': math.pi,
        'yaw_valid': yaw_valid, 'yaw_variance': math.radians(8)**2,
        'age_s': 0., 'yaw_age_s': 0., 'yaw_capture_s': t, 'track_id': 1}
    return sample


class CornerPredictor:
    def probabilities(self, features):
        return np.array([.01, .01, .98])


class OrientationContractTests(unittest.TestCase):
    def test_v3_preserves_v2_prefix_and_requires_new_version(self):
        observation = from_vision_state(pose_state(1.))
        self.assertEqual(len(observation.to_array()), 32)
        self.assertEqual(observation.opponent_yaw_valid, 1.)
        self.assertEqual(OrientationObservation.from_array(observation.to_array()), observation)
        with self.assertRaises(ValueError):
            OrientationObservation.from_array(observation.to_array()[:26])
        with self.assertRaises(ValueError):
            from_vision_state(state(1.))

    def test_yaw_requires_position_nonnegative_uncertainty_and_valid_clocks(self):
        for field, value in (('opponent_yaw_variance', -.1), ('opponent_capture_time_s', 2.),
                             ('opponent_age_s', -.1), ('opponent_position_valid', 0.)):
            observation = from_vision_state(pose_state(1.))
            values = vars(observation).copy()
            values[field] = value
            with self.assertRaises(ValueError):
                OrientationObservation(**values)


class PoseKeeperTests(unittest.TestCase):
    def test_pose_cannot_trigger_before_shot_or_override_clear_ball_side(self):
        policy = VisionPoseKeeper(predictor=CornerPredictor())
        for t in (.1, .2, .3):
            self.assertEqual(policy(pose_state(t))['trigger'], 0.)
        self.assertEqual(policy(pose_state(.4, y=-.8, vx=-1))['trigger'], 0.)
        action = policy(pose_state(.5, y=-.8, vx=-1))
        self.assertEqual(action['trigger'], 1.)
        self.assertEqual(action['block_side'], -1)
        self.assertFalse(policy.decision['pose_prior_used'])

    def test_confirmed_fresh_pose_resolves_guard_boundary_and_commit_stays_fixed(self):
        policy = VisionPoseKeeper(predictor=CornerPredictor())
        policy(pose_state(.1)); policy(pose_state(.2))
        self.assertEqual(policy(pose_state(.3, vx=-1))['trigger'], 0.)
        action = policy(pose_state(.4, vx=-1))
        self.assertEqual(action, {'block_side': 1, 'trigger': 1., 'amplitude': .5})
        self.assertTrue(policy.decision['pose_prior_used'])
        self.assertEqual(policy(pose_state(.5, y=-.8, vx=-1))['block_side'], 1)

    def test_missing_stale_or_uncertain_heading_falls_back_to_ball_v2(self):
        for condition in ('missing', 'old', 'uncertain'):
            policy = VisionPoseKeeper(predictor=CornerPredictor())
            policy(pose_state(.1)); policy(pose_state(.2))
            for t in (.3, .4):
                sample = pose_state(t, vx=-1)
                if condition == 'missing': sample['opponent']['yaw_valid'] = False
                if condition == 'old': sample['opponent']['yaw_age_s'] = .4
                if condition == 'uncertain': sample['opponent']['yaw_variance'] = .5
                action = policy(sample)
            self.assertEqual(action['block_side'], 0)
            self.assertFalse(policy.decision['pose_prior_used'])

    def test_orientation_recovery_after_commit_cannot_change_first_guard(self):
        policy = VisionPoseKeeper(predictor=CornerPredictor())
        policy(pose_state(.1)); policy(pose_state(.2))
        policy(pose_state(.3, vx=-1, yaw_valid=False))
        self.assertEqual(policy(pose_state(.4, vx=-1, yaw_valid=False))['block_side'], 0)
        self.assertEqual(policy(pose_state(.5, vx=-1, yaw_valid=True))['block_side'], 0)
        self.assertFalse(policy.decision['pose_prior_used'])

    def test_nonfinite_orientation_cannot_supply_a_prior_or_trigger(self):
        policy = VisionPoseKeeper(predictor=CornerPredictor())
        sample = pose_state(.1, vx=-1)
        sample['opponent']['yaw'] = float('nan')
        self.assertEqual(policy(sample)['trigger'], 0.)
        self.assertEqual(policy.decision['reason'], 'invalid_observation')

    def test_published_state_is_rejected_and_repeated_capture_cannot_confirm_shot(self):
        policy = VisionPoseKeeper(predictor=CornerPredictor())
        self.assertEqual(policy(state(.1))['trigger'], 0.)
        self.assertEqual(policy.decision['reason'], 'invalid_version')
        self.assertEqual(policy(pose_state(.2, vx=-1))['trigger'], 0.)
        self.assertEqual(policy(pose_state(.2, vx=-1))['trigger'], 0.)


class OwnSensorTests(unittest.TestCase):
    def test_only_own_sensordata_and_fsr_are_inputs_to_camera_kinematics(self):
        values = {name: 0. for name in ENCODERS}
        values.update(HeadPitch=-.5, LShoulderPitch=1.5, RShoulderPitch=1.5)
        calls = []
        class SensorData:
            time = 0.
            def sensor(self, name):
                calls.append(name)
                if name == 'keeper_imu_gyro': return SimpleNamespace(data=np.array([0., 0., .5]))
                if name == 'keeper_imu_acc': return SimpleNamespace(data=np.array([0., 0., 9.81]))
                self_name = name.removeprefix('keeper_encoder_')
                return SimpleNamespace(data=np.array([values[self_name]]))
        sim = SimpleNamespace(data=SensorData(), adapter=SimpleNamespace(_fsr=lambda role: [1.]*8))
        reader = OwnSensorReader(sim)
        first = copy.deepcopy(reader.read())
        values['HeadYaw'] = .3
        sim.data.time = .012
        second = reader.read()
        self.assertTrue(first['valid'])
        self.assertAlmostEqual(second['relative_yaw'], .006, places=6)
        a, b = np.asarray(first['camera_to_ground']['lower']).reshape(4, 4), np.asarray(second['camera_to_ground']['lower']).reshape(4, 4)
        self.assertAlmostEqual(math.atan2(b[1, 0], b[0, 0]), .3, places=5)
        self.assertGreater(a[2, 3], .4)
        self.assertTrue(all(name.startswith('keeper_') for name in calls))
        sim.data.time = .2
        self.assertFalse(reader.read()['valid'])


class VisualOrientationTests(unittest.TestCase):
    def setUp(self):
        if not (DEFAULT_MODEL/'templates.npz').is_file():
            self.skipTest('orientation references not built')
        self.estimator = VisualYawEstimator()
        self.rgb = np.asarray(Image.open(ROOT/'docs/validation/vision/bhuman_opponent_20261010/lower_camera.png'))
        self.box = (120, 45, 200, 184)

    def test_pixels_measure_front_heading_and_independent_captures_confirm(self):
        first = self.estimator.estimate(self.rgb, self.box, .1, .1, 1, 0.)
        second = self.estimator.estimate(self.rgb, self.box, .2, .2, 1, 0.)
        self.assertFalse(first['yaw_valid'])
        self.assertTrue(second['yaw_valid'])
        self.assertLess(abs(wrap(second['yaw']-math.pi)), math.radians(15))
        self.assertGreater(second['yaw_variance'], 0.)

    def test_missing_clipped_duplicate_stale_and_physical_domain_clear_validity(self):
        self.estimator.estimate(self.rgb, self.box, .1, .1, 1, 0.)
        self.assertFalse(self.estimator.estimate(self.rgb, self.box, .1, .1, 1, 0.)['yaw_valid'])
        self.assertFalse(self.estimator.estimate(self.rgb, self.box, .2, .6, 1, 0.)['yaw_valid'])
        self.assertFalse(self.estimator.estimate(self.rgb, self.box, .7, .7, 1, 0., source='naoqi_camera')['yaw_valid'])
        self.assertFalse(self.estimator.estimate(np.zeros_like(self.rgb), self.box, .8, .8, 1, 0.)['yaw_valid'])
        self.assertIsNone(body_feature(self.rgb, (-2, 0, 100, 100))[0])
        with self.assertRaises(ValueError):
            self.estimator.estimate(self.rgb, self.box, float('nan'), 1., 1, 0.)

    def test_equally_matching_front_and_back_are_unknown(self):
        feature, _ = body_feature(self.rgb, self.box)
        self.estimator.features = np.repeat(feature[None, :], 24, axis=0)
        self.estimator.norms = np.einsum('ij,ij->i', self.estimator.features, self.estimator.features)
        self.estimator.yaws = np.array([0., math.pi]*12)
        sample = self.estimator.estimate(self.rgb, self.box, .1, .1, 1, 0.)
        self.assertFalse(sample['yaw_valid'])
        self.assertEqual(sample['reason'], 'ambiguous_body_heading')

    def test_confirmation_compensates_own_turn_across_wrap_boundary(self):
        self.estimator.estimate(self.rgb, self.box, .1, .1, 1, 0., odometry_yaw=0.)
        # Same appearance ray, now own heading differs. Candidate local yaw
        # changes, but the measured body's odometry-frame heading stays fixed.
        sample = self.estimator.estimate(self.rgb, self.box, .2, .2, 1, -.5, odometry_yaw=.5)
        self.assertTrue(sample['yaw_valid'])

    def test_scene_helpers_are_unavailable_to_complete_sensor_camera_path(self):
        from penalty.motion_contract import executable_path
        from penalty.robot_detector import executable_path as vision_executable
        if not executable_path().is_file() or not Path(vision_executable()).is_file():
            self.skipTest('both native hosts required')
        from penalty.sim_orientation import SensorCameraDuelWorld, SensorBHumanKeeperCamera
        world, camera = SensorCameraDuelWorld(), None
        try:
            world.reset(np.random.default_rng(68000), {})
            camera = SensorBHumanKeeperCamera(world)
            for index in range(5):
                camera.observe()
                world._advance(until=lambda: world.session.penalty_time >= (index+1)*.12)
            with patch.object(world, '_ground_pose', side_effect=AssertionError('scene robot pose leaked')), \
                 patch.object(world, '_ball_xy', side_effect=AssertionError('scene ball leaked')), \
                 patch.object(world.sim, 'pose', side_effect=AssertionError('scene torso pose leaked')):
                sample = camera.observe()
            self.assertTrue(sample['ball']['position_valid'])
            self.assertTrue(sample['opponent']['yaw_valid'])
            self.assertEqual(sample['sensors']['version'], 'simulated_own_sensors_v1')
            self.assertEqual(from_vision_state(sample).opponent_yaw_valid, 1.)
        finally:
            try:
                if camera: camera.close()
            finally:
                world.close()
        self.assertIsNotNone(camera.detector.process.poll())


if __name__ == '__main__':
    unittest.main()
