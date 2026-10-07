"""World snapshot -> robot-relative RLObservation.

Ball velocity is a finite difference of successive ball estimates (what a NAO
ball model provides), not the simulator's privileged velocity. Backends state
which inputs are scene truth in `PRIVILEGED_*`; noise, delay and dropout are
added by `penalty.rl.wrappers`.
"""
from dataclasses import dataclass
import math

import numpy as np

from penalty.rl import geometry
from penalty.rl.contracts import RLObservation


@dataclass
class WorldState:
    """Ground-frame snapshot. Poses are (x, y) of the feet midpoint plus yaw."""
    time_s: float
    remaining_time: float
    ball_xy: tuple
    robot_xy: tuple
    robot_yaw: float
    opponent_xy: tuple
    opponent_yaw: float
    torso_roll: float = 0.
    torso_pitch: float = 0.
    gyro: tuple = (0., 0., 0.)
    ball_visible: bool = True


class ObservationBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self._previous = None  # (time, world ball xy) of last visible ball
        self._velocity = (0., 0.)
        self._last_seen = None
        self._ball_xy = None

    def build(self, state):
        if state.ball_visible:
            if self._previous is not None and state.time_s > self._previous[0]:
                dt = state.time_s - self._previous[0]
                self._velocity = ((state.ball_xy[0]-self._previous[1][0])/dt,
                                  (state.ball_xy[1]-self._previous[1][1])/dt)
            self._previous = (state.time_s, tuple(state.ball_xy))
            self._last_seen = state.time_s
            self._ball_xy = tuple(state.ball_xy)
        ball_xy = self._ball_xy if self._ball_xy is not None else state.ball_xy
        age = 0. if self._last_seen is None else max(0., state.time_s - self._last_seen)
        origin, yaw = state.robot_xy, state.robot_yaw
        bx, by = geometry.to_robot(ball_xy, origin, yaw)
        vx, vy = geometry.rotate_to_robot(self._velocity, yaw)
        ox, oy = geometry.to_robot(state.opponent_xy, origin, yaw)
        return RLObservation(
            time_s=state.time_s, ball_x=bx, ball_y=by, ball_vx=vx, ball_vy=vy, ball_age=age,
            opponent_x=ox, opponent_y=oy,
            opponent_yaw=geometry.wrap(state.opponent_yaw - yaw),
            goal_center_bearing=geometry.bearing(geometry.GOAL_CENTER, origin, yaw),
            left_post_bearing=geometry.bearing(geometry.LEFT_POST, origin, yaw),
            right_post_bearing=geometry.bearing(geometry.RIGHT_POST, origin, yaw),
            torso_roll=state.torso_roll, torso_pitch=state.torso_pitch,
            gyro_x=state.gyro[0], gyro_y=state.gyro[1], gyro_z=state.gyro[2],
            remaining_time=max(0., state.remaining_time),
            ball_visible=1. if state.ball_visible else 0.)


def torso_roll_pitch(rotation):
    """Roll/pitch from a 3x3 body->world rotation matrix."""
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.asin(float(np.clip(-r[2, 0], -1., 1.)))
    return roll, pitch


def ball_distance(obs):
    return math.hypot(obs.ball_x, obs.ball_y)


def ball_speed(obs):
    return math.hypot(obs.ball_vx, obs.ball_vy)
