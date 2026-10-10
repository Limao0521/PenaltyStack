import hashlib
import io
import json
import math
from pathlib import Path
import queue
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from penalty.opponent_perception import GroundCameraCalibration, OpponentFusion, ConfirmedLowerFusion
from penalty.perception import vision_state
from penalty.robot_detector import RobotDetector, SOURCE_COMMIT, executable_path, model_path
from tests.test_keeper_vision import state, transform as camera_transform

ROOT = Path(__file__).resolve().parents[1]


def transform():
    return camera_transform(pitch=0.)


def upper_calibration():
    return GroundCameraCalibration(640, 480, 600, 600, 319.5, 239.5, 'test-upper')


def box_packet(capture=.1):
    # Camera at z=.5, looking horizontally: feet at x=2 -> row 389.5.
    return {'version': 'bhuman_robot_pixels_v1', 'width': 640, 'height': 480,
            'capture_s': capture, 'boxes': [{'box': (289.5, 210., 349.5, 389.5), 'confidence': .8}]}


class OpponentGeometryTests(unittest.TestCase):
    def test_feet_projection_has_metric_position_and_nonzero_uncertainty(self):
        projection = upper_calibration().project(box_packet()['boxes'][0], transform())
        self.assertAlmostEqual(projection['position'][0], 2.)
        self.assertAlmostEqual(projection['position'][1], 0.)
        self.assertGreater(projection['position_covariance'][0][0], .0064)
        self.assertAlmostEqual(projection['field_width_m'], .2)

    def test_clipped_feet_horizon_large_or_low_confidence_boxes_are_not_range(self):
        c = upper_calibration()
        for candidate in ({'box': (290, 100, 350, 479), 'confidence': .8},
                          {'box': (290, 100, 350, 200), 'confidence': .8},
                          {'box': (0, 0, 630, 390), 'confidence': .8},
                          {'box': (290, 100, 350, 390), 'confidence': .59}):
            self.assertIsNone(c.project(candidate, transform()))

    def test_invalid_camera_transform_is_rejected(self):
        t = list(transform())
        t[0] = 2
        with self.assertRaises(ValueError):
            upper_calibration().project(box_packet()['boxes'][0], t)
        t = list(transform()); t[15] = 0
        with self.assertRaises(ValueError):
            upper_calibration().project(box_packet()['boxes'][0], t)

    def test_fusion_selects_robot_near_ball_without_fabricating_yaw(self):
        f = OpponentFusion()
        out = f.merge(state(.1, x=2.), box_packet(), upper_calibration(), transform(),
                      (0, 0, 0), (0, 0, 0), .1)
        self.assertTrue(out['opponent']['position_valid'])
        self.assertFalse(out['opponent']['yaw_valid'])
        self.assertEqual(out['opponent']['type'], 'unknown')
        self.assertEqual(out['opponent_detector']['reason'], 'selected')
        self.assertEqual(out['opponent']['track_id'], 1)

    def test_fusion_compensates_own_odometry_and_rotates_covariance(self):
        f = OpponentFusion()
        out = f.merge(state(.1, x=1.5, y=.5), box_packet(), upper_calibration(), transform(),
                      (0, 0, 0), (.5, -.5, 0), .1)
        self.assertAlmostEqual(out['opponent']['position'][0], 1.5)
        self.assertAlmostEqual(out['opponent']['position'][1], .5)
        next_packet = box_packet(.2)
        out = f.merge(state(.2, x=.5, y=-1.5), next_packet, upper_calibration(), transform(),
                      (0, 0, 0), (.5, -.5, math.pi/2), .2)
        self.assertAlmostEqual(out['opponent']['position'][0], .5)
        self.assertAlmostEqual(out['opponent']['position'][1], -1.5)
        self.assertEqual(out['opponent']['track_id'], 1)

    def test_duplicate_stale_unsynchronized_missing_or_far_robot_clears_masks(self):
        f = OpponentFusion()
        good = f.merge(state(.1, x=2.), box_packet(), upper_calibration(), transform(),
                       (0, 0, 0), (0, 0, 0), .1)
        duplicate = f.merge(good, box_packet(), upper_calibration(), transform(),
                            (0, 0, 0), (0, 0, 0), .1)
        self.assertFalse(duplicate['opponent']['position_valid'])
        self.assertEqual(duplicate['opponent_detector']['reason'], 'duplicate_or_out_of_order')
        for sample, capture, sync in ((state(.5, x=2), .2, True), (state(.6, x=2), .6, False),
                                       (state(.7, x=4), .7, True)):
            out = f.merge(sample, box_packet(capture), upper_calibration(), transform(),
                          (0, 0, 0), (0, 0, 0), capture, synchronized=sync)
            self.assertFalse(out['opponent']['position_valid'])
            self.assertFalse(out['opponent']['yaw_valid'])
        missing = state(.8, x=2); missing['ball']['position_valid'] = False
        out = f.merge(missing, box_packet(.8), upper_calibration(), transform(), (0, 0, 0), (0, 0, 0), .8)
        self.assertFalse(out['opponent']['position_valid'])

    def test_future_frame_or_wrong_calibration_resolution_is_rejected(self):
        f = OpponentFusion()
        with self.assertRaises(ValueError):
            f.merge(state(.1), box_packet(.2), upper_calibration(), transform(), (0, 0, 0), (0, 0, 0), .1)
        p = box_packet(); p['width'] = 320
        with self.assertRaises(ValueError):
            f.merge(state(.1), p, upper_calibration(), transform(), (0, 0, 0), (0, 0, 0), .1)

    def test_unscored_lower_candidates_require_two_distinct_consistent_captures(self):
        fusion = ConfirmedLowerFusion()
        for stamp, expected in ((.1, False), (.2, True), (.6, False), (.7, True)):
            packet = box_packet(stamp)
            packet['boxes'][0]['confidence_measured'] = False
            out = fusion.merge(state(stamp, x=2), packet, upper_calibration(), transform(), (0, 0, 0), (0, 0, 0), stamp)
            self.assertEqual(out['opponent']['position_valid'], expected)
            self.assertIsNone(out['opponent']['confidence'])
            self.assertFalse(out['opponent']['yaw_valid'])
        duplicate = fusion.merge(state(.7, x=2), box_packet(.7), upper_calibration(), transform(),
                                 (0, 0, 0), (0, 0, 0), .7)
        self.assertFalse(duplicate['opponent']['position_valid'])

    def test_original_import_and_generated_bodies_have_verified_provenance(self):
        original = ROOT / 'heritage/bhuman_robot_perception'
        manifest = json.loads((original / 'manifest.json').read_text())
        self.assertEqual(manifest['commit'], SOURCE_COMMIT)
        for name, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((original / name).read_bytes()).hexdigest(), expected, name)
        from tools.build_robot_perception import prepare
        provenance = prepare()
        generated = ROOT / 'native/bhuman_motion/build/generated/vision/original_algorithms.h'
        self.assertEqual(hashlib.sha256(generated.read_bytes()).hexdigest(), provenance['generated_algorithms_sha256'])


class RobotNativeTests(unittest.TestCase):
    def setUp(self):
        if not Path(executable_path()).is_file():
            self.skipTest('build tools/build_robot_perception.py before native image tests')

    def test_actual_network_negative_frame_and_yuyv_input(self):
        detector = RobotDetector()
        try:
            self.assertFalse(detector.hello['actuators'])
            out = detector.detect(bytes(640*480*3), 640, 480, 0.)
            self.assertEqual(out['boxes'], [])
            self.assertLess(out['max_confidence'], .6)
            yuyv = bytes([0, 128, 0, 128])*(320*240//2)
            out = detector.detect(yuyv, 320, 240, .1, pixel_format='yuyv')
            self.assertEqual(out['boxes'], [])
            self.assertEqual(out['sequence'], 1)
        finally:
            detector.close()
        self.assertIsNotNone(detector.process.poll())
        self.assertTrue(all(not thread.is_alive() for thread in detector._threads))

    def fake_child(self, code, timeout=.03):
        manifest = json.loads((ROOT / 'heritage/bhuman_robot_perception/manifest.json').read_text())
        hello = {'type': 'HELLO', 'protocol': 'bhuman_robot_pixels_v1', 'source_commit': SOURCE_COMMIT,
                 'model_sha256': manifest['files']['Config/NeuralNets/RobotDetector/4_anchor_boxes_model_no_activation_20230629-220730.onnx'],
                 'actuators': False}
        script = 'import json, sys, time\nprint(%r, flush=True)\n' % json.dumps(hello) + code
        return RobotDetector(command=[sys.executable, '-u', '-c', script], timeout=timeout)

    def test_timeout_reaps_nonconsuming_child(self):
        detector = self.fake_child('time.sleep(30)\n')
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            detector.detect(bytes(640*480*3), 640, 480, 0.)
        self.assertLess(time.monotonic()-start, 1.5)
        self.assertIsNotNone(detector.process.poll())
        self.assertTrue(all(not thread.is_alive() for thread in detector._threads))

    def test_malformed_output_closes_child(self):
        detector = self.fake_child('sys.stdin.readline()\nprint("[]", flush=True)\ntime.sleep(30)\n', timeout=.2)
        with self.assertRaises(RuntimeError):
            detector.detect(bytes(320*240*3), 320, 240, 0.)
        self.assertIsNotNone(detector.process.poll())

    def test_full_queue_does_not_skip_process_collection(self):
        detector = self.fake_child('time.sleep(30)\n')
        try:
            for _ in range(20):
                detector._outgoing.put_nowait(b'x'*1200000)
        except queue.Full:
            pass
        detector.close()
        self.assertIsNotNone(detector.process.poll())
        self.assertTrue(all(not thread.is_alive() for thread in detector._threads))

    def test_actual_camera_path_never_reads_opponent_or_ball_scene_coordinates(self):
        from penalty.motion_contract import executable_path as motion_executable
        if not motion_executable().is_file():
            self.skipTest('motion runtime required for rendered pixels')
        from penalty.sim_opponent_vision import BHumanCameraDuelWorld, BHumanKeeperCamera
        world, camera = BHumanCameraDuelWorld(), None
        try:
            world.reset(np.random.default_rng(66000), {})
            camera = BHumanKeeperCamera(world)
            world._advance(until=lambda: world.session.penalty_time >= 1.2)
            camera.observe()  # First lower contrast candidate is unconfirmed.
            world._advance(until=lambda: world.session.penalty_time >= 1.32)
            own_sensing = world._ground_pose
            def own_only(role):
                self.assertEqual(role, 'keeper')
                return own_sensing(role)
            with patch.object(world, '_ball_xy', side_effect=AssertionError('privileged ball leaked')), \
                 patch.object(world, '_ground_pose', side_effect=own_only):
                packet = camera.observe()
            self.assertTrue(packet['ball']['position_valid'])
            self.assertTrue(packet['opponent']['position_valid'])
            self.assertFalse(packet['opponent']['yaw_valid'])
            self.assertIn('boxes', packet['opponent_detector'])
            self.assertEqual(packet['runtime_provenance']['source_commit'], SOURCE_COMMIT)
        finally:
            try:
                if camera: camera.close()
            finally:
                world.close()


class NaoOpponentTests(unittest.TestCase):
    @staticmethod
    def profile():
        return {'role': 'keeper', 'keeper_policy': 'vision', 'camera': 1,
                'vision': {'calibration': {'width': 320, 'height': 240, 'fx': 300, 'fy': 300,
                    'cx': 159.5, 'cy': 119.5, 'ball_radius_m': .0325, 'calibration_id': 'test-lower'},
                    'opponent': {'detector': 'bhuman', 'calibration': {'width': 640, 'height': 480,
                        'fx': 600, 'fy': 600, 'cx': 319.5, 'cy': 239.5, 'calibration_id': 'test-upper'}}}}

    def test_no_actuators_and_independent_camera_times_with_start_abort(self):
        from penalty.nao_opponent_vision import run
        class Clock:
            current = 0.
            def __call__(self):
                self.current += .01
                return self.current
        clock = Clock()
        class Memory:
            def getData(self, key):
                return 1. if 'Head/Touch' in key or clock.current >= .7 else 0.
        class Video:
            unsubscribed = []
            def subscribeCamera(self, name, index, resolution, color, fps):
                self.assertions = (index, resolution, color, fps)
                return name
            def getImageRemote(self, name):
                w, h = (320, 240) if name == 'PenaltyStackBall' else (640, 480)
                stamp = 1000.+clock.current-.01
                return [w, h, 3, 11, int(stamp), int((stamp-int(stamp))*1e6), bytes(w*h*3)]
            def unsubscribe(self, camera):
                self.unsubscribed.append(camera)
        class Motion:
            def getTransform(self, name, frame, sensors):
                return transform()
            def getRobotPosition(self, sensors):
                return (0, 0, 0)
            def __getattr__(self, name):
                raise AssertionError('actuator call: '+name)
        class Detector:
            hello = {'actuators': False, 'source_commit': SOURCE_COMMIT}
            closed = False
            def detect(self, pixels, width, height, capture, camera='upper'):
                packet = box_packet(capture)
                packet.update(width=width, height=height, camera=camera)
                return packet
            def close(self):
                self.closed = True
        video, detector = Video(), Detector()
        class Session:
            def service(self, name):
                return {'ALMemory': Memory(), 'ALVideoDevice': video, 'ALMotion': Motion()}[name]
        output = io.StringIO()
        with patch('penalty.nao_opponent_vision.time.time', side_effect=lambda: 1000.+clock.current):
            run(self.profile(), Session(), clock=clock, sleeper=lambda _: None, output=output, detector=detector)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(any(r.get('active') for r in rows))
        self.assertEqual(rows[-1]['reason'], 'abort')
        self.assertEqual(len(video.unsubscribed), 2)
        self.assertTrue(detector.closed)
        for row in rows[:-1]:
            self.assertFalse(row['executed'])
            self.assertFalse(row['vision']['opponent']['yaw_valid'])
            self.assertNotEqual(row['vision']['capture_s'], row['vision']['opponent_detector_upper']['capture_s'])

    def test_native_observer_profile_refuses_physical_moves(self):
        from penalty.nao_opponent_vision import validate_profile
        validate_profile(self.profile())
        for flag in ('motion_enabled', 'kick_enabled', 'block_enabled'):
            with self.assertRaises(ValueError):
                validate_profile(dict(self.profile(), **{flag: True}))

    def test_camera_and_log_cleanup_failures_still_reap_detector(self):
        from unittest.mock import MagicMock
        from penalty.nao_opponent_vision import run
        memory, video, detector = MagicMock(), MagicMock(), MagicMock()
        memory.getData.side_effect = RuntimeError('observer stopped')
        video.subscribeCamera.side_effect = ['ball', 'upper']
        video.unsubscribe.side_effect = RuntimeError('camera channel closed')
        session = MagicMock()
        session.service.side_effect = lambda name: {'ALMemory': memory, 'ALVideoDevice': video,
                                                    'ALMotion': MagicMock()}[name]
        output = io.StringIO(); output.close()
        with self.assertRaisesRegex(RuntimeError, 'observer stopped'):
            run(self.profile(), session=session, detector=detector, output=output)
        self.assertEqual(video.unsubscribe.call_count, 2)
        detector.close.assert_called_once()

    def test_package_refuses_missing_linux_runtime_and_windows_binary(self):
        from deploy.package import package
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = dict(self.profile(), robot_id='NAO_TEST_SERIAL')
            path = root / 'profile.json'
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, 'Linux'):
                package(path, root / 'out.tar.gz')
            windows = root / 'bad.exe'; windows.write_bytes(b'MZ'+bytes(20))
            with self.assertRaisesRegex(ValueError, 'ELF'):
                package(path, root / 'out.tar.gz', windows)


if __name__ == '__main__':
    unittest.main()
