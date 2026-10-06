"""Adaptador onboard NAOqi 2.8. Python 2.7 compatible.

El modo inicial observa y registra. La marcha exige activación explícita en
el perfil y en el entorno; la patada física todavía no tiene ejecutor.
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys
import time

from penalty.control import Observation, PenaltyController
from penalty.vision import detect_orange_ball

START_KEY = 'Device/SubDeviceList/Head/Touch/Front/Sensor/Value'
ABORT_KEY = 'Device/SubDeviceList/ChestBoard/Button/Sensor/Value'


def held(memory, key, seconds, now, since):
    pressed = float(memory.getData(key)) > 0.5
    if not pressed:
        return False, None
    if since is None:
        since = now
    return now - since >= seconds, since


def run(profile_path):
    with open(profile_path, 'r') as stream:
        profile = json.load(stream)
    role = profile['role']
    if role not in ('striker', 'keeper'):
        raise ValueError('invalid role')
    motion_enabled = bool(profile.get('motion_enabled', False))
    if motion_enabled and os.environ.get('PENALTY_ENABLE_WALK') != 'I_UNDERSTAND':
        raise RuntimeError('walk requires PENALTY_ENABLE_WALK=I_UNDERSTAND')
    if profile.get('kick_enabled', False):
        raise RuntimeError('physical kick executor is not integrated')

    import qi  # Import only on the robot.
    session = qi.Session()
    session.connect('tcp://127.0.0.1:9559')
    memory = session.service('ALMemory')
    video = session.service('ALVideoDevice')
    motion = session.service('ALMotion')
    # 1 = bottom camera, 1 = QVGA, 11 = RGB, 10 FPS.
    camera = video.subscribeCamera('PenaltyStack', int(profile.get('camera', 1)), 1, 11, 10)
    controller = PenaltyController(role)
    start_since = None
    abort_since = None
    started = False
    epoch = time.time()
    previous_state = None
    last_head = 0.0
    last_frame_id = None
    last_frame_received = None
    try:
        if motion_enabled:
            motion.wakeUp()
            motion.moveInit()
        while True:
            now = time.time()
            started_now, start_since = held(memory, START_KEY, 0.25, now, start_since)
            aborted, abort_since = held(memory, ABORT_KEY, 0.3, now, abort_since)
            if started_now:
                started = True
            image = video.getImageRemote(camera)
            ball = None
            ball_age = 1.0
            if image and image[6]:
                ball = detect_orange_ball(image[6], int(image[0]), int(image[1]))
                if len(image) > 5:
                    frame_id = (image[4], image[5])
                    if frame_id != last_frame_id:
                        last_frame_id = frame_id
                        last_frame_received = now
                    ball_age = now - last_frame_received
            obs = Observation(now - epoch, start=started, abort=aborted,
                              ball_range=ball[0] if ball else None,
                              ball_bearing=ball[1] if ball else None,
                              ball_age=ball_age,
                              goal_bearing=float(profile.get('goal_bearing_rad', 0.0)))
            command = controller.update(obs)
            if command.state != previous_state:
                print('%.2f %s ball=%s' % (obs.now, command.state, ball))
                previous_state = command.state
            if motion_enabled:
                # NAOqi uses normalized velocities. These provisional reference
                # speeds must be measured on the actual robot before competition.
                motion.moveToward(max(-0.18, min(0.18, command.forward / 0.15)),
                                  max(-0.12, min(0.12, command.lateral / 0.08)),
                                  max(-0.12, min(0.12, command.turn / 0.6)))
                if abs(command.head_yaw - last_head) > 0.08:
                    motion.setAngles('HeadYaw', command.head_yaw, 0.08)
                    last_head = command.head_yaw
            if command.kick:
                raise RuntimeError('KICK_REQUEST_NOT_EXECUTED: physical executor missing')
            if command.state == 'STOP':
                break
            time.sleep(0.1)
    finally:
        if motion_enabled:
            motion.stopMove()
        video.unsubscribe(camera)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    run(args.profile)


if __name__ == '__main__':
    main()
