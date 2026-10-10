"""Synthetic keeper sensor stress v2, separate from published noise v1.

Adds yaw/velocity uncertainty and one tactical frame of latency. This is
an analytical stress test on scene-derived observations, not a calibrated
camera model or evidence of hardware readiness.
"""
from collections import deque
import math

import numpy as np

from penalty.rl.contracts import FIELD, clip_observation
from penalty.rl.wrappers import PerceptionNoise


class KeeperSensorStressV2:
    version = 'keeper_sensor_stress_v2'

    def __init__(self, yaw_sigma=.15, velocity_sigma=.06, delay_steps=1):
        if (not all(math.isfinite(v) and 0 <= v <= 1 for v in (yaw_sigma, velocity_sigma))
                or type(delay_steps) is not int or not 0 <= delay_steps <= 3):
            raise ValueError('ruido finito en [0,1] y retraso entero en [0,3] requeridos')
        self.yaw_sigma, self.velocity_sigma, self.delay_steps = yaw_sigma, velocity_sigma, delay_steps
        self.base = PerceptionNoise(dropout=.1)
        self.frames = deque(maxlen=delay_steps+1)

    def __call__(self, observation, rng):
        value = self.base(observation, rng)
        yaw = float(value[FIELD['opponent_yaw']]) + rng.normal(0., self.yaw_sigma)
        value[FIELD['opponent_yaw']] = (yaw+math.pi) % (2*math.pi)-math.pi
        value[[FIELD['ball_vx'], FIELD['ball_vy']]] += rng.normal(0., self.velocity_sigma, 2)
        self.frames.append(value.copy())
        delayed = self.frames[0].copy()
        now = float(value[FIELD['time_s']])
        age = max(0., now-float(delayed[FIELD['time_s']]))
        delayed[FIELD['time_s']] = now
        delayed[FIELD['remaining_time']] = value[FIELD['remaining_time']]
        delayed[FIELD['ball_age']] += age
        return clip_observation(delayed)
