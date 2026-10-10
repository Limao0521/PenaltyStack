"""Temporal keeper baseline; uses only rl_obs_v1 and keeper_action_v1.

Pose supplies a pre-shot prior. Ball observations confirm the trajectory;
the original B-Human guard is requested when it can cover the arriving ball.
No simulator contact event, opponent identity or target is available here.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np

from penalty.rl.anticipation import CLASSES, KickSidePredictor, pose_features
from penalty.rl.contracts import BLOCK_CENTER, BLOCK_LEFT, BLOCK_RIGHT, FIELD, KeeperAction
from penalty.rl.geometry import predict_goal_crossing


@dataclass(frozen=True)
class RobustKeeperSettings:
    corner_y: float = .3
    dive_lead_s: float = 1.8
    deceleration: float = 0.
    min_incoming_speed: float = .12
    confirm_frames: int = 2
    history: int = 3
    max_ball_age: float = .25
    pose_threshold: float = .9

    def __post_init__(self):
        bounds = {'corner_y': (.2, .6), 'dive_lead_s': (.3, 3.),
                  'deceleration': (0., .6), 'min_incoming_speed': (.05, .4),
                  'max_ball_age': (0., .5), 'pose_threshold': (.5, 1.)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError('%s fuera de [%g, %g]' % (name, low, high))
        for name in ('confirm_frames', 'history'):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 6:
                raise ValueError('%s debe ser entero entre 1 y 6' % name)


class RobustKeeperPolicy:
    name = 'robust_keeper_v1'

    def __init__(self, settings=None, predictor=None):
        self.settings = (settings if isinstance(settings, RobustKeeperSettings) else
                         RobustKeeperSettings(**(settings or {})))
        self.predictor = predictor or KickSidePredictor.load()
        self.reset()

    def reset(self):
        self.pose_history = deque(maxlen=6)
        self.trajectories = deque(maxlen=self.settings.history)
        self.incoming_frames = 0
        self.shot_started = None
        self.committed = None
        self.last_time = None
        self.decision = {'reason': 'waiting_for_ball'}

    def describe(self):
        return {'policy': self.name, 'settings': asdict(self.settings),
                'inputs': 'rl_obs_v1 only; pose and ball estimates remain simulated'}

    def _action(self, side=BLOCK_CENTER, fire=False):
        return KeeperAction(side, float(fire), .5).to_normalized()

    def __call__(self, observation):
        o = np.asarray(observation, dtype=float)
        if o.shape != (len(FIELD),) or not np.isfinite(o).all():
            self.decision = {'reason': 'invalid_observation'}
            return self._action()
        now = float(o[FIELD['time_s']])
        if self.last_time is not None and now < self.last_time:
            self.reset()
        if self.committed is not None:
            return self._action(self.committed, True)
        if self.last_time == now:
            return self._action()
        self.last_time = now
        if o[FIELD['ball_visible']] < .5 or o[FIELD['ball_age']] > self.settings.max_ball_age:
            self.incoming_frames = 0
            self.decision = {'reason': 'stale_ball'}
            return self._action()
        bx, by, vx, vy = (float(o[FIELD[n]]) for n in ('ball_x', 'ball_y', 'ball_vx', 'ball_vy'))
        incoming = bx > 0. and vx < -self.settings.min_incoming_speed
        if not incoming:
            self.incoming_frames = 0
            if self.shot_started is None:
                self.pose_history.append(self.predictor.probabilities(pose_features(o)))
            self.decision = {'reason': 'waiting_for_shot'}
            return self._action()
        self.incoming_frames += 1
        if self.shot_started is None:
            self.shot_started = now
        prediction = predict_goal_crossing((bx, by), (vx, vy), goal_line_x=-.12,
                                            deceleration=self.settings.deceleration)
        if prediction is not None:
            self.trajectories.append((prediction['predicted_goal_y'], prediction['time_to_goal']))
        if self.incoming_frames < self.settings.confirm_frames or not self.trajectories:
            self.decision = {'reason': 'confirming_trajectory'}
            return self._action()
        y, arrival = np.median(np.asarray(self.trajectories), axis=0)
        side = BLOCK_LEFT if y > self.settings.corner_y else BLOCK_RIGHT if y < -self.settings.corner_y else BLOCK_CENTER
        # A stable pre-shot prior resolves only the uncertainty near the edge
        # of the central guard; clear ball trajectories always take precedence.
        if self.pose_history and abs(abs(y) - self.settings.corner_y) < .04:
            prior = np.mean(self.pose_history, axis=0)
            guess = int(prior.argmax())
            predicted_side = CLASSES[guess]
            if prior[guess] >= self.settings.pose_threshold and predicted_side * y > 0:
                side = predicted_side
        self.decision = {'reason': 'waiting_for_arrival', 'crossing_y': float(y),
                         'time_to_line': float(arrival), 'side': int(side)}
        # The central guard is sustained; lateral jumps have a finite window.
        if side != BLOCK_CENTER and arrival > self.settings.dive_lead_s:
            return self._action()
        self.committed = side
        self.decision['reason'] = 'committed'
        return self._action(side, True)
