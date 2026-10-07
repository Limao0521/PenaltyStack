"""Versioned observation/action contracts for the tactical RL layer.

Frames: every position is robot-relative (x forward, y left, metres) and
every angle is in radians, positive to the left. The tactical policy never
commands the 23 servos; B-Human remains the low-level controller.
"""
from dataclasses import astuple, dataclass, fields
import math

import numpy as np

OBSERVATION_VERSION = 'rl_obs_v1'
STRIKER_ACTION_VERSION = 'striker_action_v1'
KEEPER_ACTION_VERSION = 'keeper_action_v1'

KICK_LEFT, KICK_RIGHT = 0, 1
KICK_TYPES = {KICK_LEFT: 'forwardFastLeft', KICK_RIGHT: 'forwardFastRight'}
BLOCK_RIGHT, BLOCK_CENTER, BLOCK_LEFT = -1, 0, 1
BLOCK_SIDES = {BLOCK_RIGHT: 'right', BLOCK_CENTER: 'center', BLOCK_LEFT: 'left'}
# Inside the posts (0.75 m) minus ball radius and a small margin.
TARGET_Y_LIMIT = 0.65


def _require_finite(owner, values):
    for name, value in values:
        if not math.isfinite(value):
            raise ValueError('%s.%s debe ser finito' % (owner, name))


@dataclass(frozen=True)
class RLObservation:
    time_s: float
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    ball_age: float
    opponent_x: float
    opponent_y: float
    opponent_yaw: float
    goal_center_bearing: float
    left_post_bearing: float
    right_post_bearing: float
    torso_roll: float
    torso_pitch: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    remaining_time: float
    # Visibility mask: 0 while the ball is not perceived; ball_x/y then hold
    # the last estimate and ball_age grows.
    ball_visible: float = 1.0

    def __post_init__(self):
        _require_finite('RLObservation', ((f.name, getattr(self, f.name)) for f in fields(self)))

    def to_array(self):
        return np.asarray(astuple(self), dtype=np.float32)

    @classmethod
    def from_array(cls, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(OBSERVATION_FIELDS),):
            raise ValueError('observación debe tener %d valores' % len(OBSERVATION_FIELDS))
        return cls(*(float(v) for v in values))


OBSERVATION_FIELDS = tuple(f.name for f in fields(RLObservation))
FIELD = {name: index for index, name in enumerate(OBSERVATION_FIELDS)}
_PI = math.pi
_BOUNDS = {
    'time_s': (0., 40.), 'ball_x': (-6., 6.), 'ball_y': (-6., 6.),
    'ball_vx': (-12., 12.), 'ball_vy': (-12., 12.), 'ball_age': (0., 40.),
    'opponent_x': (-6., 6.), 'opponent_y': (-6., 6.), 'opponent_yaw': (-_PI, _PI),
    'goal_center_bearing': (-_PI, _PI), 'left_post_bearing': (-_PI, _PI),
    'right_post_bearing': (-_PI, _PI), 'torso_roll': (-_PI, _PI), 'torso_pitch': (-_PI, _PI),
    'gyro_x': (-30., 30.), 'gyro_y': (-30., 30.), 'gyro_z': (-30., 30.),
    'remaining_time': (0., 28.), 'ball_visible': (0., 1.),
}
OBS_LOW = np.array([_BOUNDS[n][0] for n in OBSERVATION_FIELDS], dtype=np.float32)
OBS_HIGH = np.array([_BOUNDS[n][1] for n in OBSERVATION_FIELDS], dtype=np.float32)


def clip_observation(values):
    return np.clip(np.asarray(values, dtype=np.float32), OBS_LOW, OBS_HIGH)


def _unit(value, name):
    if not 0. <= value <= 1.:
        raise ValueError('%s debe estar en [0, 1]' % name)


def _normalized(values, size):
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError('acción normalizada debe tener %d valores finitos' % size)
    return np.clip(values, -1., 1.)


@dataclass(frozen=True)
class StrikerAction:
    """target_y: goal-line aim point in metres, positive to the striker's left."""
    target_y: float
    kick_side: int
    kick_power: float
    trigger: float

    def __post_init__(self):
        _require_finite('StrikerAction', (('target_y', self.target_y),
                        ('kick_power', self.kick_power), ('trigger', self.trigger)))
        if abs(self.target_y) > TARGET_Y_LIMIT:
            raise ValueError('target_y fuera de la portería')
        if self.kick_side not in KICK_TYPES:
            raise ValueError('kick_side debe ser KICK_LEFT o KICK_RIGHT')
        _unit(self.kick_power, 'kick_power')
        _unit(self.trigger, 'trigger')

    @property
    def fires(self):
        return self.trigger >= .5

    @classmethod
    def from_normalized(cls, values):
        a = _normalized(values, 4)
        return cls(float(a[0])*TARGET_Y_LIMIT, KICK_RIGHT if a[1] > 0 else KICK_LEFT,
                   float(a[2]+1)/2, float(a[3]+1)/2)

    def to_normalized(self):
        return np.array([self.target_y/TARGET_Y_LIMIT, 1. if self.kick_side == KICK_RIGHT else -1.,
                         self.kick_power*2-1, self.trigger*2-1], dtype=np.float32)


@dataclass(frozen=True)
class KeeperAction:
    """WAIT is trigger < 0.5; otherwise block_side is committed once."""
    block_side: int
    trigger: float
    amplitude: float

    def __post_init__(self):
        _require_finite('KeeperAction', (('trigger', self.trigger), ('amplitude', self.amplitude)))
        if self.block_side not in BLOCK_SIDES:
            raise ValueError('block_side debe ser BLOCK_LEFT, BLOCK_CENTER o BLOCK_RIGHT')
        _unit(self.trigger, 'trigger')
        _unit(self.amplitude, 'amplitude')

    @property
    def fires(self):
        return self.trigger >= .5

    @classmethod
    def from_normalized(cls, values):
        a = _normalized(values, 3)
        side = BLOCK_LEFT if a[0] > 1/3 else BLOCK_RIGHT if a[0] < -1/3 else BLOCK_CENTER
        return cls(side, float(a[1]+1)/2, float(a[2]+1)/2)

    def to_normalized(self):
        return np.array([self.block_side*2/3, self.trigger*2-1, self.amplitude*2-1],
                        dtype=np.float32)


def versions():
    return {'observation': OBSERVATION_VERSION, 'striker_action': STRIKER_ACTION_VERSION,
            'keeper_action': KEEPER_ACTION_VERSION}
