"""Vision geometry, clocks, odometry, missing information and keeper decisions."""
import io
import math
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from penalty.keeper_vision import VisionKeeper
from penalty.perception import (BHumanVisionAdapter, BallTracker, CameraCalibration, CameraVision,
                                ShooterSelector, orange_candidate, vision_state)
from penalty.rl.contracts import KeeperAction, OBSERVATION_FIELDS as V1_FIELDS
from penalty.rl.vision_contracts import VisionObservation, from_vision_state, OBSERVATION_FIELDS
from penalty.rl.vision_keeper import VisionKeeperPolicy


def transform(pitch=.4, x=0., z=.5):
    c, s = math.cos(pitch), math.sin(pitch)
    return [c, 0, s, x, 0, 1, 0, 0, -s, 0, c, z, 0, 0, 0, 1]


def measurement(x=1., y=0., variance=1e-6):
    return {'position': (x, y), 'covariance': ((variance, 0), (0, variance))}


def state(t, x=1., y=0., vx=-1., vy=0., capture=None):
    b = {'position': (x, y), 'velocity': (vx, vy), 'position_valid': True,
         'velocity_valid': True, 'visible': True, 'age_s': 0., 'last_seen_s': t,
         'position_covariance': ((1e-6, 0), (0, 1e-6)),
         'velocity_covariance': ((0., 0.), (0., 0.))}
    return vision_state('simulated_camera', t if capture is None else capture, t, t, t, b)


class CameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cal = CameraCalibration(320, 240, 300, 300, 159.5, 119.5, .0325, 'test')

    def test_projection_uses_camera_pose_and_left_positive(self):
        a = self.cal.project_ball({'center': (159.5, 119.5), 'diameter': 16.}, transform())
        b = self.cal.project_ball({'center': (140, 119.5), 'diameter': 16.}, transform())
        self.assertAlmostEqual(a['position'][0], (.5-.0325)/math.tan(.4))
        self.assertAlmostEqual(a['position'][1], 0.)
        self.assertGreater(b['position'][1], 0.)
        self.assertGreater(a['covariance'][0][0], 0.)

    def test_rejects_upward_ray_incorrect_diameter_and_nonrigid_transform(self):
        candidate = {'center': (159.5, 119.5), 'diameter': 16.}
        self.assertIsNone(self.cal.project_ball(candidate, transform(-.4)))
        self.assertIsNone(self.cal.project_ball(dict(candidate, diameter=100.), transform()))
        bad = transform()
        bad[0] = 8
        with self.assertRaises(ValueError):
            self.cal.project_ball(candidate, bad)

    def test_detector_rejects_clipped_blob_and_preserves_centroid(self):
        def image(cx):
            rgb = np.zeros((60, 80, 3), dtype=np.uint8)
            yy, xx = np.indices((60, 80))
            rgb[(xx-cx)**2+(yy-30)**2 <= 6**2] = (230, 110, 20)
            return rgb.tobytes()
        candidate = orange_candidate(image(40), 80, 60, stride=1)
        self.assertEqual(candidate['center'], (40., 30.))
        self.assertIsNone(orange_candidate(image(0), 80, 60, stride=1))

    def test_unsynchronized_image_has_no_valid_measurement(self):
        vision = CameraVision(self.cal, 'simulated_camera')
        packet = vision.process(None, 0., 0., .1, transform(), (0, 0, 0), (0, 0, 0), .1,
                                synchronized=False)
        self.assertFalse(packet['ball']['position_valid'])
        self.assertEqual(packet['ball']['reason'], 'unsynchronized_sensors')


class BallTrackingTests(unittest.TestCase):
    def test_one_frame_no_velocity_then_ego_translation_and_yaw_do_not_move_ball(self):
        tracker = BallTracker()
        tracker.update(0., 0., measurement(2., 0.), (0., 0., 0.))
        self.assertFalse(tracker.estimate(0., (0, 0, 0))['velocity_valid'])
        tracker.update(.1, .1, measurement(0., -1.8), (.2, 0., math.pi/2))
        b = tracker.estimate(.1, (.2, 0., math.pi/2))
        np.testing.assert_allclose(b['velocity'], (0., 0.), atol=1e-10)
        self.assertTrue(b['velocity_valid'])

    def test_true_incoming_velocity_and_capture_latency(self):
        tracker = BallTracker()
        tracker.update(0., 0., measurement(1.), (0, 0, 0))
        tracker.update(.1, .2, measurement(.9), (0, 0, 0))
        b = tracker.estimate(.2, (0, 0, 0))
        self.assertAlmostEqual(b['velocity'][0], -1.)
        self.assertAlmostEqual(b['position'][0], .8)
        self.assertAlmostEqual(b['age_s'], .1)

    def test_duplicates_out_of_order_occlusion_gap_and_stale_frame(self):
        tracker = BallTracker()
        tracker.update(0., 0., measurement(), (0, 0, 0))
        self.assertFalse(tracker.update(0., .1, measurement(.8), (0, 0, 0)))
        self.assertFalse(tracker.update(-.1, .1, measurement(.8), (0, 0, 0)))
        tracker.update(.1, .1, None, (0, 0, 0))
        self.assertFalse(tracker.estimate(.1, (0, 0, 0))['visible'])
        self.assertFalse(tracker.estimate(.3, (0, 0, 0))['position_valid'])
        tracker.update(.4, .4, measurement(.7), (0, 0, 0))
        self.assertFalse(tracker.estimate(.4, (0, 0, 0))['velocity_valid'])
        self.assertFalse(tracker.update(.5, 1., measurement(.6), (0, 0, 0)))
        with self.assertRaises(ValueError):
            tracker.update(2., 1., measurement(), (0, 0, 0))

    def test_outlier_does_not_poison_next_velocity(self):
        tracker = BallTracker()
        tracker.update(0, 0, measurement(), (0, 0, 0))
        self.assertFalse(tracker.update(.1, .1, measurement(3.), (0, 0, 0)))
        tracker.update(.2, .2, measurement(.8), (0, 0, 0))
        self.assertFalse(tracker.estimate(.2, (0, 0, 0))['velocity_valid'])


class ModelAdapterTests(unittest.TestCase):
    def packet(self):
        return {'version': 'bhuman_local_models_v1', 'source': 'bhuman_models', 'capture_s': 1.,
                'odometry': (0, 0, 0), 'ball_model': {
                    'position_mm': (1000, 100), 'velocity_mm_s': (-500, 50),
                    'covariance_mm2': ((100, 0), (0, 100)),
                    'velocity_covariance_mm2_s2': ((100, 0), (0, 100)),
                    'last_seen_s': 1., 'valid': True, 'velocity_valid': True, 'seen_this_frame': True},
                'obstacles': [{'center_mm': (1200, 100), 'covariance_mm2': ((100, 0), (0, 100)),
                               'last_seen_s': 1., 'type': 'opponent', 'velocity_mm_ms': (0., 1.)}]}

    def test_units_validity_and_no_yaw_inference(self):
        adapter = BHumanVisionAdapter()
        packet = self.packet()
        state = adapter.process(packet, 1., 1., 1.)
        self.assertEqual(state['ball']['position'], (1., .1))
        self.assertEqual(state['ball']['velocity'], (-.5, .05))
        self.assertAlmostEqual(state['ball']['position_covariance'][0][0], .0001)
        self.assertTrue(state['opponent']['position_valid'])
        self.assertFalse(state['opponent']['yaw_valid'])
        self.assertFalse(adapter.process(packet, 1.1, 1.1, 1.1)['ball']['velocity_valid'])

    def test_model_latency_odometry_and_missing_velocity_covariance(self):
        packet = self.packet()
        state = BHumanVisionAdapter().process(packet, 1.1, 1.1, 1.1, current_odometry=(.1, 0., 0.))
        self.assertAlmostEqual(state['ball']['position'][0], .85)
        packet['ball_model']['velocity_covariance_mm2_s2'] = None
        state = BHumanVisionAdapter().process(packet, 1., 1., 1.)
        self.assertFalse(state['ball']['velocity_valid'])
        self.assertTrue(state['ball']['position_valid'])

    def test_future_clock_rejected_without_poisoning_adapter(self):
        adapter, packet = BHumanVisionAdapter(), self.packet()
        with self.assertRaises(ValueError):
            adapter.process(packet, .5, .5, .5)
        self.assertTrue(adapter.process(packet, 1., 1., 1.)['ball']['position_valid'])

    def test_selector_survives_list_reordering_and_rejects_teammate(self):
        selector = ShooterSelector()
        opponent = {'position': (1.2, 0), 'type': 'opponent', 'last_seen_s': 0.}
        teammate = dict(opponent, position=(1., 0), type='teammate')
        first = selector.select([teammate, opponent], (1., 0), 0., (0, 0, 0))
        second = selector.select([opponent, teammate], (1., 0), .1, (0, 0, 0))
        self.assertEqual(first['track_id'], second['track_id'])
        self.assertEqual(first['position'], (1.2, 0))
        self.assertIsNone(selector.select([opponent], (1., 0), 1., (0, 0, 0)))


class VisionKeeperTests(unittest.TestCase):
    def test_v2_preserves_v1_schema_and_requires_masks(self):
        self.assertEqual(len(V1_FIELDS), 19)
        self.assertEqual(OBSERVATION_FIELDS[:19], V1_FIELDS)
        policy = VisionKeeperPolicy()
        with self.assertRaises(ValueError):
            policy(np.zeros(19))
        array = from_vision_state(state(.1)).to_array()
        np.testing.assert_array_equal(VisionObservation.from_array(array).to_array(), array)

    def test_confirm_independent_captures_and_decide_without_opponent(self):
        for vy, side in ((.5, 1), (-.5, -1), (0, 0)):
            keeper = VisionKeeper()
            first = state(.1, vy=vy)
            self.assertEqual(keeper(first)['trigger'], 0)
            duplicate = state(.15, vy=vy, capture=.1)
            self.assertEqual(keeper(duplicate)['trigger'], 0)
            self.assertEqual(keeper(state(.2, vy=vy))['block_side'], side)
            self.assertEqual(keeper.decision['reason'], 'committed')
            self.assertFalse(keeper.decision['pose_prior_used'])

    def test_missing_velocity_stale_ball_and_uncertain_speed_cannot_commit(self):
        for key in ('velocity_valid', 'position_valid', 'visible'):
            keeper = VisionKeeper()
            sample = state(.1)
            sample['ball'][key] = False
            self.assertEqual(keeper(sample)['trigger'], 0)
        keeper = VisionKeeper()
        for t in (.1, .2, .3):
            sample = state(t, vx=-.2)
            sample['ball']['velocity_covariance'] = ((.1, 0), (0, .1))
            self.assertEqual(keeper(sample)['trigger'], 0)
        self.assertEqual(keeper.decision['reason'], 'waiting_for_shot')

    def test_loss_clears_trajectory_and_deadline_disables_committed_action(self):
        keeper = VisionKeeper()
        keeper(state(.1))
        loss = state(.2)
        loss['ball']['visible'] = False
        keeper(loss)
        self.assertEqual(keeper(state(.3))['trigger'], 0)
        self.assertEqual(keeper(state(.4))['trigger'], 1)
        self.assertEqual(keeper(state(28.))['trigger'], 0)

    def test_portable_and_array_policies_produce_the_same_action(self):
        core, array = VisionKeeper(), VisionKeeperPolicy()
        for t in (.1, .2, .3):
            packet = state(t, vy=.5)
            self.assertEqual(KeeperAction(**core(packet)),
                             KeeperAction.from_normalized(array(from_vision_state(packet).to_array())))

    def test_replay_and_standard_factory_use_the_versioned_policy(self):
        from tools.replay_keeper_vision import replay
        from penalty.rl.policies.baselines import make_policy
        self.assertIsInstance(make_policy('vision', 'keeper'), VisionKeeperPolicy)
        rows = list(replay([{'vision': state(.1)}, {'vision': state(.2)}]))
        self.assertEqual([r['action']['trigger'] for r in rows], [0., 1.])
        self.assertTrue(all(r['executed'] is False for r in rows))

    def test_onboard_mode_refuses_actuator_enable(self):
        from penalty.nao_vision import validate_profile
        profile = {'role': 'keeper', 'keeper_policy': 'vision', 'vision': {'calibration': {
            'width': 320, 'height': 240, 'fx': 300, 'fy': 300, 'cx': 159.5, 'cy': 119.5,
            'ball_radius_m': .0325, 'calibration_id': 'test'}}}
        validate_profile(profile)
        with self.assertRaises(ValueError):
            validate_profile(dict(profile, block_enabled=True))

    def test_onboard_start_abort_cleanup_and_no_actuator_calls(self):
        from penalty.nao_vision import run
        class Clock:
            current = 0.
            def __call__(self):
                self.current += .01
                return self.current
        clock = Clock()
        class Memory:
            def getData(self, key):
                return 1. if 'Head/Touch' in key or clock.current >= .6 else 0.
        class Video:
            unsubscribed = False
            def subscribeCamera(self, *args):
                return 'test-camera'
            def getImageRemote(self, camera):
                stamp = 1000.+clock.current-.01
                return [320, 240, 3, 11, int(stamp), int((stamp-int(stamp))*1e6), bytes(320*240*3)]
            def unsubscribe(self, camera):
                self.unsubscribed = True
        class Motion:
            def getTransform(self, name, frame, sensors):
                return transform()
            def getRobotPosition(self, sensors):
                return (0, 0, 0)
            def __getattr__(self, name):
                raise AssertionError('actuator call: '+name)
        video = Video()
        class Session:
            def service(self, name):
                return {'ALMemory': Memory(), 'ALVideoDevice': video, 'ALMotion': Motion()}[name]
        profile = {'role': 'keeper', 'keeper_policy': 'vision', 'vision': {'calibration': {
            'width': 320, 'height': 240, 'fx': 300, 'fy': 300, 'cx': 159.5, 'cy': 119.5,
            'ball_radius_m': .0325, 'calibration_id': 'test'}}}
        output = io.StringIO()
        with patch('penalty.nao_vision.time.time', side_effect=lambda: 1000.+clock.current):
            run(profile, Session(), clock=clock, sleeper=lambda _: None, output=output)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(any(row.get('active') for row in rows))
        self.assertEqual(rows[-1]['reason'], 'abort')
        self.assertTrue(video.unsubscribed)
        self.assertTrue(all(row['executed'] is False for row in rows[:-1]))

    def test_camera_controller_ignores_privileged_ball(self):
        from penalty.control import Observation
        from penalty.sim_vision import CameraKeeperController
        keeper = CameraKeeperController()
        command = keeper.update(Observation(0., start=True, ball_range=.1, ball_bearing=-1.))
        self.assertEqual(command.head_yaw, 0.)
        keeper.look(state(.1, y=.4))
        command = keeper.update(Observation(.1, start=True, ball_range=.1, ball_bearing=-1.))
        self.assertGreater(command.head_yaw, 0.)
        self.assertFalse(command.block_now)


class CameraAcceptanceTests(unittest.TestCase):
    def test_renderer_cleanup_failure_still_closes_native_world(self):
        from penalty.rl.vision_benchmark import _job
        world, camera = MagicMock(), MagicMock()
        camera.model = world.sim.model
        camera.close.side_effect = RuntimeError('GL cleanup failure')
        with patch('penalty.rl.vision_benchmark.CameraDuelWorld', return_value=world), \
             patch('penalty.rl.vision_benchmark.SimulatedKeeperCamera', return_value=camera), \
             patch('penalty.rl.vision_benchmark.camera_episode', return_value={'result': 'save'}):
            with self.assertRaisesRegex(RuntimeError, 'GL cleanup'):
                _job({'keeper': 'vision', 'style': 'center', 'seeds': [1], 'trained': {}})
        world.close.assert_called_once()

    def test_rendered_camera_observes_ball_without_access_to_scene_ball_or_opponent(self):
        from penalty.motion_contract import executable_path
        if not executable_path().is_file():
            self.skipTest('native motion runtime unavailable')
        from penalty.sim_vision import CameraDuelWorld, SimulatedKeeperCamera
        world, camera = CameraDuelWorld(), None
        try:
            world.reset(np.random.default_rng(64002), {})
            camera = SimulatedKeeperCamera(world)
            # Original B-Human head engine moves into its camera-view request.
            world._advance(until=lambda: world.session.penalty_time >= .5)
            with patch.object(world, '_ball_xy', side_effect=AssertionError('scene ball leaked')):
                packet = camera.observe()
            self.assertEqual(packet['source'], 'simulated_camera')
            self.assertTrue(packet['ball']['visible'])
            self.assertFalse(packet['opponent']['position_valid'])
            self.assertFalse(packet['opponent']['yaw_valid'])
            self.assertAlmostEqual(packet['ball']['position'][0], 1.18, delta=.15)
        finally:
            if camera:
                camera.close()
            world.close()


if __name__ == '__main__':
    unittest.main()
