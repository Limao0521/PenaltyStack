"""Onboard camera keeper in observation mode only; Python 2.7 compatible.

No actuator commands: logs the same portable v2 decision used in simulation.
NAOqi transforms/odometry are read from sensors. No B-Human providers start
here, and an absent opponent detector stays explicitly invalid.
"""
from __future__ import division, print_function

import json
import sys
import time

from penalty.keeper_vision import POLICY_VERSION, VisionKeeper
from penalty.perception import CameraCalibration, CameraVision


def monotonic_seconds():
    if hasattr(time, 'monotonic'):
        return time.monotonic()
    # NAOqi 2.8 ships Python 2.7 on Linux, which lacks time.monotonic.
    import ctypes
    class Timespec(ctypes.Structure):
        _fields_ = [('seconds', ctypes.c_long), ('nanoseconds', ctypes.c_long)]
    value = Timespec()
    library = ctypes.CDLL('librt.so.1', use_errno=True)
    if library.clock_gettime(1, ctypes.byref(value)) != 0:  # CLOCK_MONOTONIC
        raise RuntimeError('monotonic clock unavailable')
    return value.seconds + value.nanoseconds*1e-9


def validate_profile(profile):
    if profile.get('role') != 'keeper' or profile.get('keeper_policy') != 'vision':
        raise ValueError('vision keeper profile required')
    if any(profile.get(key, False) for key in ('motion_enabled', 'kick_enabled', 'block_enabled')):
        raise ValueError('vision keeper is observation-only; physical block executor is uncalibrated')
    calibration = CameraCalibration(**profile['vision']['calibration'])
    if 'TO_FILL' in calibration.calibration_id:
        raise ValueError('measure camera calibration and set calibration_id')
    if (calibration.width, calibration.height) != (320, 240):
        raise ValueError('onboard observer currently requires calibrated QVGA RGB')
    if profile.get('camera', 1) not in (0, 1):
        raise ValueError('camera must be 0 (upper) or 1 (lower)')
    VisionKeeper(**profile.get('vision', {}).get('keeper_settings', {}))
    return calibration


def run(profile, session=None, clock=monotonic_seconds, sleeper=time.sleep, output=None):
    from penalty.nao import START_KEY, ABORT_KEY, held
    calibration = validate_profile(profile)
    if session is None:
        import qi
        session = qi.Session()
        session.connect('tcp://127.0.0.1:9559')
    memory, video, motion = (session.service(name) for name in ('ALMemory', 'ALVideoDevice', 'ALMotion'))
    camera_index = int(profile.get('camera', 1))
    camera_name = 'CameraBottom' if camera_index == 1 else 'CameraTop'
    camera = video.subscribeCamera('PenaltyStackVision', camera_index, 1, 11, 10)
    vision = CameraVision(calibration, 'naoqi_camera')
    keeper = VisionKeeper(**profile.get('vision', {}).get('keeper_settings', {}))
    output = output or sys.stdout
    start_since = abort_since = started_s = None
    wall_offset = time.time()-clock()
    last_clock = None
    try:
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
                keeper.reset()
            elapsed = 0. if started_s is None else now-started_s
            if aborted or (started_s is not None and elapsed >= 28.):
                output.write(json.dumps({'event': 'STOP', 'reason': 'abort' if aborted else 'deadline',
                                         'time_s': elapsed, 'policy': POLICY_VERSION})+'\n')
                output.flush()
                break
            image = video.getImageRemote(camera)
            received = clock()
            if abs(time.time()-received-wall_offset) > .02:
                raise RuntimeError('wall/camera clock changed; re-establish capture clock')
            if not image or not image[6]:
                sleeper(.02)
                continue
            if (int(image[0]), int(image[1])) != (calibration.width, calibration.height):
                raise ValueError('camera resolution differs from calibration')
            capture = float(image[4])+float(image[5])*1e-6-wall_offset
            # Read-only sensor calls, never motion/wakeUp/stiffness requests.
            transform = motion.getTransform(camera_name, 2, True)  # FRAME_ROBOT
            odometry = motion.getRobotPosition(True)
            sensor_time = clock()
            synchronized = 0 <= sensor_time-capture <= .08
            elapsed = 0. if started_s is None else sensor_time-started_s
            state = vision.process(image[6], capture, received, sensor_time, transform,
                                   odometry, odometry, elapsed, synchronized=synchronized)
            state['sensor_provenance'] = 'NAOqi measured transform/odometry; receipt-time approximation'
            state['sensor_time_s'] = sensor_time
            action = keeper(state) if started_s is not None else {'block_side': 0, 'trigger': 0., 'amplitude': .5}
            if elapsed >= 28.:
                action['trigger'] = 0.
            output.write(json.dumps({'vision': state, 'policy': POLICY_VERSION,
                                     'active': started_s is not None, 'action': action,
                                     'decision': keeper.decision, 'executed': False}, allow_nan=False)+'\n')
            output.flush()
            sleeper(.02)
    finally:
        video.unsubscribe(camera)
