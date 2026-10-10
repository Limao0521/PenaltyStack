"""Kick-side anticipation from the striker's pose, as the keeper observes it.

Before the kick the tactical striker orbits the ball to aim, so its position
relative to the ball and its heading reveal the side. On B-Human corner
kicks with start variation (2026-10-10) a multinomial logistic regression on
these features predicted the side of 77 % of 571 kicks (87 % of corner
kicks), and already 82 % of corner kicks two seconds before the kick.

AnticipatingKeeperPolicy waits until the ball moves and then dives to the
predicted corner when the predictor is confident, centre guard otherwise.
"""
import json
import math
from pathlib import Path

import numpy as np

from penalty.motion_contract import ROOT
from penalty.rl.contracts import BLOCK_CENTER, BLOCK_LEFT, BLOCK_RIGHT, FIELD, KeeperAction

DEFAULT_PREDICTOR = ROOT/'models'/'rl'/'kick_side_predictor.json'
CLASSES = (BLOCK_RIGHT, BLOCK_CENTER, BLOCK_LEFT)  # keeper's view
SIDE_Y = .2
SHOT_SPEED = .15  # m/s: the ball has been kicked
FEATURES = ('ball_x', 'ball_y', 'striker_dx', 'striker_dy', 'sin_yaw', 'cos_yaw', 'pointing_y')


def pose_features(observation):
    """Striker pose in the keeper frame: offset from the ball, heading, and
    the y where its heading line from the ball meets the keeper's line."""
    o = np.asarray(observation, dtype=np.float64)
    bx, by = o[FIELD['ball_x']], o[FIELD['ball_y']]
    ox, oy, yaw = o[FIELD['opponent_x']], o[FIELD['opponent_y']], o[FIELD['opponent_yaw']]
    hx, hy = math.cos(yaw), math.sin(yaw)
    pointing = by + hy*(-bx/hx) if hx < -1e-3 else by
    return np.array([bx, by, ox - bx, oy - by, math.sin(yaw), math.cos(yaw), pointing])


def side_class(world_y):
    """Index into CLASSES of where the ball crossed (keeper's left is world +y)."""
    return 2 if world_y > SIDE_Y else 0 if world_y < -SIDE_Y else 1


class KickSidePredictor:
    def __init__(self, mean, std, weights, metadata=None):
        self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)
        self.weights = np.asarray(weights, float)
        self.metadata = dict(metadata or {})

    @classmethod
    def fit(cls, features, labels, l2=1e-2, steps=4000, rate=.5, metadata=None):
        x, y = np.asarray(features, float), np.asarray(labels, int)
        mean, std = x.mean(0), x.std(0) + 1e-9
        design = np.hstack([(x - mean)/std, np.ones((len(x), 1))])
        weights = np.zeros((design.shape[1], len(CLASSES)))
        onehot = np.eye(len(CLASSES))[y]
        for _ in range(steps):
            p = _softmax(design @ weights)
            weights -= rate*(design.T @ (p - onehot)/len(design) + l2*weights)
        return cls(mean, std, weights, metadata)

    def probabilities(self, features):
        x = (np.atleast_2d(features) - self.mean)/self.std
        p = _softmax(np.hstack([x, np.ones((len(x), 1))]) @ self.weights)
        return p[0] if np.ndim(features) == 1 else p

    def save(self, path):
        Path(path).write_text(json.dumps({'features': FEATURES, 'classes': CLASSES, 'mean': self.mean.tolist(),
                                          'std': self.std.tolist(), 'weights': self.weights.tolist(),
                                          'metadata': self.metadata}, indent=1) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path=DEFAULT_PREDICTOR):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if tuple(data['features']) != FEATURES or tuple(data['classes']) != CLASSES:
            raise ValueError('predictor de lado incompatible: %s' % path)
        return cls(data['mean'], data['std'], data['weights'], data.get('metadata'))


def _softmax(logits):
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p/p.sum(axis=1, keepdims=True)


class AnticipatingKeeperPolicy:
    """Dive to the predicted corner when P >= threshold, centre guard
    otherwise; commit only once the ball moves (no filter needed)."""
    name = 'anticipating_keeper'

    def __init__(self, predictor=None, threshold=.75):
        self.predictor = predictor or KickSidePredictor.load()
        self.threshold = threshold

    def __call__(self, observation):
        o = np.asarray(observation)
        moving = math.hypot(o[FIELD['ball_vx']], o[FIELD['ball_vy']]) > SHOT_SPEED
        if not moving:
            return KeeperAction(BLOCK_CENTER, 0., .5).to_normalized()
        p = self.predictor.probabilities(pose_features(o))
        guess = int(p.argmax())
        side = CLASSES[guess] if CLASSES[guess] != BLOCK_CENTER and p[guess] >= self.threshold else BLOCK_CENTER
        return KeeperAction(side, 1., .5).to_normalized()
