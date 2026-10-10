"""Ball-only temporal keeper v2 for local perception; Python 2.7 compatible.

Shares the timing idea of robust_keeper_v1, with independent captures,
validity/covariance checks and measured capture latency. No opponent yaw prior
is used until a body-orientation estimator has been validated.
"""
from __future__ import division

from collections import deque
import math

from penalty.perception import VISION_VERSION, finite

POLICY_VERSION = 'vision_keeper_v2'


def median(values):
    values = sorted(values)
    mid = len(values)//2
    return values[mid] if len(values) % 2 else (values[mid-1]+values[mid])/2.


class VisionKeeper(object):
    def __init__(self, corner_y=.3, dive_lead_s=1.8, min_incoming_speed=.12,
                 confirm_frames=2, history=3, max_ball_age=.25, sigma_margin=1.5,
                 goal_line_x=-.12):
        self.corner_y, self.dive_lead_s = corner_y, dive_lead_s
        self.min_incoming_speed, self.max_ball_age = min_incoming_speed, max_ball_age
        self.confirm_frames, self.history = confirm_frames, history
        self.sigma_margin, self.goal_line_x = sigma_margin, goal_line_x
        for value, low, high in ((corner_y, .2, .6), (dive_lead_s, .3, 3.),
                                  (min_incoming_speed, .05, .5), (max_ball_age, .01, .5),
                                  (sigma_margin, 0., 4.), (goal_line_x, -.5, 0.)):
            if not finite(value) or not low <= value <= high:
                raise ValueError('invalid vision keeper configuration')
        if any(type(v) is not int or not 1 <= v <= 6 for v in (confirm_frames, history)):
            raise ValueError('confirmation/history must be integers in [1, 6]')
        self.reset()

    def reset(self):
        self.last_capture = self.last_time = self.committed = None
        self.incoming_frames = 0
        self.trajectories = deque(maxlen=self.history)
        self.decision = {'reason': 'waiting_for_ball', 'pose_prior_used': False}

    def wait(self, reason):
        self.incoming_frames = 0
        self.trajectories.clear()
        self.decision = {'reason': reason, 'pose_prior_used': False}
        return {'block_side': 0, 'trigger': 0., 'amplitude': .5}

    def __call__(self, state):
        if state.get('version') != VISION_VERSION:
            return self.wait('invalid_version')
        now, capture = state['time_s'], state['capture_s']
        if not all(finite(v) for v in (now, capture, state['remaining_time'])):
            return self.wait('invalid_observation')
        if self.last_time is not None and now < self.last_time:
            self.reset()
        self.last_time = now
        if state['remaining_time'] <= 0 or now >= 28:
            return self.wait('deadline')
        if self.committed is not None:
            return {'block_side': self.committed, 'trigger': 1., 'amplitude': .5}
        if self.last_capture is not None and capture <= self.last_capture:
            return {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        self.last_capture = capture
        ball = state['ball']
        if (not ball['position_valid'] or not ball['visible'] or not 0 <= ball['age_s'] <= self.max_ball_age):
            return self.wait('stale_ball')
        if not ball['velocity_valid']:
            return self.wait('unmeasured_velocity')
        bx, by = ball['position']
        vx, vy = ball['velocity']
        variance = ball['velocity_covariance'][0][0]
        if not all(finite(v) for v in (bx, by, vx, vy, variance)) or variance < 0:
            return self.wait('invalid_ball')
        threshold = self.min_incoming_speed + self.sigma_margin*math.sqrt(variance)
        if bx <= 0 or vx >= -threshold:
            return self.wait('waiting_for_shot')
        self.incoming_frames += 1
        arrival = (self.goal_line_x-bx)/vx
        y = by+vy*arrival
        self.trajectories.append((y, arrival))
        if self.incoming_frames < self.confirm_frames:
            self.decision = {'reason': 'confirming_trajectory', 'pose_prior_used': False}
            return {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        y = median([p[0] for p in self.trajectories])
        arrival = median([p[1] for p in self.trajectories])
        side = 1 if y > self.corner_y else -1 if y < -self.corner_y else 0
        self.decision = {'reason': 'waiting_for_arrival', 'crossing_y': y,
                         'time_to_line': arrival, 'side': side, 'pose_prior_used': False}
        if side != 0 and arrival > self.dive_lead_s:
            return {'block_side': side, 'trigger': 0., 'amplitude': .5}
        self.committed = side
        self.decision['reason'] = 'committed'
        return {'block_side': side, 'trigger': 1., 'amplitude': .5}
