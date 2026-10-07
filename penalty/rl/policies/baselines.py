"""Baseline tactical policies on raw RLObservation arrays.

Each policy maps one observation (np.ndarray, OBSERVATION_FIELDS order) to a
normalized action for the corresponding env, so baselines and learned
policies are evaluated through the same interface.
"""
import math

import numpy as np

from penalty.rl.contracts import (BLOCK_CENTER, BLOCK_LEFT, BLOCK_RIGHT, FIELD, KICK_LEFT,
                                  KeeperAction, RLObservation, StrikerAction)
from penalty.rl.geometry import predict_goal_crossing
from penalty.skills.kick import NOMINAL_POWER

KEEPER_LINE_TO_GOAL = .12  # keeper stands 0.12 m in front of its goal line


class ScriptedStrikerPolicy:
    """Same decision as the shared FSM: centre, left foot, 2500 mm, at once."""
    name = 'scripted_striker'

    def __init__(self, target_y=0., kick_side=KICK_LEFT, kick_power=NOMINAL_POWER):
        self.action = StrikerAction(target_y, kick_side, kick_power, 1.).to_normalized()

    def __call__(self, observation):
        return self.action.copy()


class ScriptedKeeperPolicy:
    """Same gate as SimulationSession: approaching ball, side by bearing."""
    name = 'scripted_keeper'

    def __call__(self, observation):
        o = RLObservation.from_array(observation)
        distance, speed = math.hypot(o.ball_x, o.ball_y), math.hypot(o.ball_vx, o.ball_vy)
        bearing = math.atan2(o.ball_y, o.ball_x)
        side = BLOCK_LEFT if bearing > .18 else BLOCK_RIGHT if bearing < -.18 else BLOCK_CENTER
        fire = speed > .15 and distance < 1.1
        return KeeperAction(side, 1. if fire else 0., .5).to_normalized()


class GeometricKeeperPolicy:
    """Commits once the predicted crossing is close in time."""
    name = 'geometric_keeper'

    def __init__(self, commit_time=.7, side_threshold=.2, deceleration=.6):
        self.commit_time, self.side_threshold = commit_time, side_threshold
        self.deceleration = deceleration

    def __call__(self, observation):
        o = observation
        prediction = predict_goal_crossing((o[FIELD['ball_x']], o[FIELD['ball_y']]),
                                           (o[FIELD['ball_vx']], o[FIELD['ball_vy']]),
                                           goal_line_x=-KEEPER_LINE_TO_GOAL,
                                           deceleration=self.deceleration)
        if prediction is None or prediction['time_to_goal'] > self.commit_time:
            return KeeperAction(BLOCK_CENTER, 0., .5).to_normalized()
        y = prediction['predicted_goal_y']
        side = (BLOCK_LEFT if y > self.side_threshold else
                BLOCK_RIGHT if y < -self.side_threshold else BLOCK_CENTER)
        return KeeperAction(side, 1., 1.).to_normalized()


class RandomPolicy:
    name = 'random'

    def __init__(self, action_size, seed=None):
        self.size = action_size
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation):
        return self.rng.uniform(-1., 1., self.size).astype(np.float32)


def make_policy(name, role, seed=None):
    if role == 'striker' and name == 'scripted':
        return ScriptedStrikerPolicy()
    if role == 'keeper' and name == 'scripted':
        return ScriptedKeeperPolicy()
    if role == 'keeper' and name == 'geometric':
        return GeometricKeeperPolicy()
    if name == 'random':
        return RandomPolicy(4 if role == 'striker' else 3, seed)
    raise ValueError('política desconocida para %s: %s' % (role, name))
