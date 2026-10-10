"""rl_obs_v2: v1 fields plus explicit validity, capture time and uncertainty.

Published rl_obs_v1 and all policies/checkpoints consuming it stay unchanged.
Zero placeholders for unknown opponents are usable only with their masks.
Provenance and full covariances remain in the accompanying vision_state_v1.
"""
from dataclasses import astuple, dataclass, fields

import numpy as np

from penalty.rl.contracts import RLObservation, _require_finite

OBSERVATION_VERSION = 'rl_obs_v2'


@dataclass(frozen=True)
class VisionObservation(RLObservation):
    ball_position_valid: float = 0.
    ball_velocity_valid: float = 0.
    opponent_position_valid: float = 0.
    opponent_yaw_valid: float = 0.
    capture_time_s: float = 0.
    decision_time_s: float = 0.
    ball_velocity_x_variance: float = 0.

    def __post_init__(self):
        _require_finite('VisionObservation', ((f.name, getattr(self, f.name)) for f in fields(self)))
        for name in ('ball_visible', 'ball_position_valid', 'ball_velocity_valid',
                     'opponent_position_valid', 'opponent_yaw_valid'):
            if getattr(self, name) not in (0., 1.):
                raise ValueError('%s must be a binary validity mask' % name)
        if (self.ball_age < 0 or self.ball_velocity_x_variance < 0 or
                self.capture_time_s > self.decision_time_s or
                self.ball_velocity_valid > self.ball_position_valid or
                self.opponent_yaw_valid > self.opponent_position_valid):
            raise ValueError('inconsistent vision observation')

    def to_array(self):
        # Absolute monotonic times require float64 to retain independent frames.
        return np.asarray(astuple(self), dtype=np.float64)

    @classmethod
    def from_array(cls, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(OBSERVATION_FIELDS),):
            raise ValueError('vision requires rl_obs_v2 (%d fields); v1 is not interchangeable' % len(OBSERVATION_FIELDS))
        return cls(*(float(v) for v in values))


OBSERVATION_FIELDS = tuple(f.name for f in fields(VisionObservation))
FIELD = {name: i for i, name in enumerate(OBSERVATION_FIELDS)}


def from_vision_state(state):
    if state['version'] != 'vision_state_v1':
        raise ValueError('vision_state_v1 required')
    b, o, sensors = state['ball'], state['opponent'], state['sensors']
    gyro = sensors.get('gyro', (0., 0., 0.))
    return VisionObservation(
        time_s=state['time_s'], remaining_time=state['remaining_time'],
        ball_x=b['position'][0], ball_y=b['position'][1],
        ball_vx=b['velocity'][0], ball_vy=b['velocity'][1], ball_age=b['age_s'],
        ball_visible=float(b['visible']), ball_position_valid=float(b['position_valid']),
        ball_velocity_valid=float(b['velocity_valid']),
        opponent_x=o['position'][0], opponent_y=o['position'][1], opponent_yaw=o['yaw'],
        opponent_position_valid=float(o['position_valid']), opponent_yaw_valid=float(o['yaw_valid']),
        goal_center_bearing=sensors.get('goal_center_bearing', 0.),
        left_post_bearing=sensors.get('left_post_bearing', 0.),
        right_post_bearing=sensors.get('right_post_bearing', 0.),
        torso_roll=sensors.get('torso_roll', 0.), torso_pitch=sensors.get('torso_pitch', 0.),
        gyro_x=gyro[0], gyro_y=gyro[1], gyro_z=gyro[2],
        capture_time_s=state['capture_s'], decision_time_s=state['decision_s'],
        ball_velocity_x_variance=b['velocity_covariance'][0][0])
