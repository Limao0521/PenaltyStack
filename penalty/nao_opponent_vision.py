"""Onboard dual-camera B-Human opponent perception, observation-only/Python 2.7."""
from __future__ import division, print_function

import json
import os
import sys
import time

from penalty.keeper_vision import POLICY_VERSION, VisionKeeper
from penalty.nao_vision import monotonic_seconds, validate_profile as validate_ball_profile
from penalty.opponent_perception import (GroundCameraCalibration, OpponentFusion,
                                        ConfirmedLowerFusion, merge_camera_results)
from penalty.perception import CameraVision
from penalty.robot_detector import ROOT, RobotDetector


def validate_profile(profile):
    ball = validate_ball_profile(profile)
    opponent = profile['vision'].get('opponent', {})
    if opponent.get('detector') != 'bhuman':
        raise ValueError('original B-Human opponent detector must be selected')
    upper = GroundCameraCalibration(**opponent['calibration'])
    if 'TO_FILL' in upper.calibration_id:
        raise ValueError('measure upper-camera calibration before observing')
    if (upper.width, upper.height) != (640, 480):
        raise ValueError('onboard robot detector requires calibrated upper VGA')
    if profile.get('camera', 1) != 1:
        raise ValueError('dual-camera observer requires lower QVGA for ball and upper VGA for robots')
    return ball, upper


def run(profile, session=None, clock=monotonic_seconds, sleeper=time.sleep, output=None, detector=None):
    from penalty.nao import START_KEY, ABORT_KEY, held
    ball_calibration, upper_calibration = validate_profile(profile)
    if session is None:
        import qi
        session = qi.Session()
        session.connect('tcp://127.0.0.1:9559')
    memory, video, motion = (session.service(name) for name in ('ALMemory', 'ALVideoDevice', 'ALMotion'))
    vision, fusion = CameraVision(ball_calibration, 'naoqi_camera'), OpponentFusion()
    lower_fusion = ConfirmedLowerFusion()
    lower_fusion.selector = fusion.selector
    lower_calibration = GroundCameraCalibration(ball_calibration.width, ball_calibration.height,
        ball_calibration.fx, ball_calibration.fy, ball_calibration.cx, ball_calibration.cy,
        ball_calibration.calibration_id + '_robot')
    keeper = VisionKeeper(**profile['vision'].get('keeper_settings', {}))
    output = output or sys.stdout
    cameras = []
    start_since = abort_since = started_s = last_clock = None
    wall_offset = time.time()-clock()
    detector_failed = None

    def sample(token, calibration, camera_name):
        image = video.getImageRemote(token)
        received = clock()
        if abs(time.time()-received-wall_offset) > .02:
            raise RuntimeError('wall/camera clock changed; restart observation')
        if not image or not image[6]:
            return None
        if (int(image[0]), int(image[1])) != (calibration.width, calibration.height):
            raise ValueError('camera resolution differs from calibration')
        capture = float(image[4])+float(image[5])*1e-6-wall_offset
        if capture > received+1e-6:
            raise ValueError('future camera timestamp')
        transform = motion.getTransform(camera_name, 2, True)
        pose = motion.getRobotPosition(True)
        sensor_time = clock()
        return {'pixels': image[6], 'capture_s': capture, 'received_s': received,
                'transform': transform, 'odometry': pose, 'sensor_time_s': sensor_time,
                'synchronized': 0 <= sensor_time-capture <= .08}

    try:
        if detector is None:
            executable = os.path.join(ROOT, 'native/bhuman_vision/bhuman_vision')
            detector = RobotDetector(executable=executable)
        ball_index = int(profile.get('camera', 1))
        ball_token = video.subscribeCamera('PenaltyStackBall', ball_index, 1, 11, 10)
        cameras.append(ball_token)
        upper_token = video.subscribeCamera('PenaltyStackOpponent', 0, 2, 11, 10)
        cameras.append(upper_token)
        while True:
            now = clock()
            if last_clock is not None and now < last_clock:
                raise RuntimeError('monotonic clock rewind')
            last_clock = now
            started, start_since = held(memory, START_KEY, .25, now, start_since)
            aborted, abort_since = held(memory, ABORT_KEY, .3, now, abort_since)
            if started and started_s is None:
                started_s = now
                vision.reset()
                fusion.reset()
                lower_fusion.reset()
                keeper.reset()
            elapsed = 0. if started_s is None else now-started_s
            if aborted or (started_s is not None and elapsed >= 28.):
                output.write(json.dumps({'event': 'STOP', 'reason': 'abort' if aborted else 'deadline',
                                         'time_s': elapsed, 'policy': POLICY_VERSION})+'\n')
                output.flush()
                break
            ball = sample(ball_token, ball_calibration, 'CameraBottom' if ball_index else 'CameraTop')
            upper = sample(upper_token, upper_calibration, 'CameraTop')
            if ball is None:
                sleeper(.02)
                continue
            detections = lower_detections = None
            if detector_failed is None:
                try:
                    lower_detections = detector.detect(ball['pixels'], 320, 240, ball['capture_s'], camera='lower')
                    if upper is not None:
                        detections = detector.detect(upper['pixels'], 640, 480, upper['capture_s'])
                except (RuntimeError, OSError, IOError) as error:
                    # A detector failure never fabricates a rival or blocks ball tracking.
                    detector_failed = str(error)
                    detector.close()
            decision = clock()
            current_pose = motion.getRobotPosition(True)
            elapsed = 0. if started_s is None else decision-started_s
            state = vision.process(ball['pixels'], ball['capture_s'], ball['received_s'], decision,
                                   ball['transform'], ball['odometry'], current_pose, elapsed,
                                   synchronized=ball['synchronized'])
            if detections is not None:
                state = fusion.merge(state, detections, upper_calibration, upper['transform'],
                                     upper['odometry'], current_pose, upper['received_s'],
                                     synchronized=upper['synchronized'])
            else:
                state['opponent_detector'] = {'reason': detector_failed or 'upper_camera_missing',
                                              'position_valid': False, 'yaw_valid': False}
            if lower_detections is not None:
                lower = lower_fusion.merge(state, lower_detections, lower_calibration, ball['transform'],
                                           ball['odometry'], current_pose, ball['received_s'],
                                           synchronized=ball['synchronized'])
                state = merge_camera_results(state, lower)
            state['camera_pipeline'] = 'keeper_bhuman_dual_naoqi_v1'
            state['runtime_provenance'] = detector.hello
            state['sensor_provenance'] = 'NAOqi measured transforms/odometry, receipt-time approximation per camera'
            state['camera_sensor_time_s'] = {'ball': ball['sensor_time_s'],
                                             'opponent': upper['sensor_time_s'] if upper else None}
            action = keeper(state) if started_s is not None else {'block_side': 0, 'trigger': 0., 'amplitude': .5}
            if elapsed >= 28.:
                action['trigger'] = 0.
            output.write(json.dumps({'vision': state, 'policy': POLICY_VERSION, 'active': started_s is not None,
                                     'action': action, 'decision': keeper.decision, 'executed': False}, allow_nan=False)+'\n')
            output.flush()
            sleeper(.02)
    finally:
        try:
            for camera in cameras:
                try:
                    video.unsubscribe(camera)
                except Exception as error:
                    try:
                        output.write(json.dumps({'event': 'CAMERA_CLOSE_ERROR', 'message': str(error)})+'\n')
                    except (IOError, OSError, ValueError):
                        pass
        finally:
            if detector is not None:
                detector.close()
