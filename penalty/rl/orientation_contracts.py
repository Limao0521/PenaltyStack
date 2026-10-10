"""rl_obs_v3: camera body yaw with age and uncertainty, separate from v2."""
from dataclasses import astuple, dataclass, fields

import numpy as np

from penalty.rl.vision_contracts import VisionObservation, from_vision_state as from_v1

OBSERVATION_VERSION = 'rl_obs_v3'


@dataclass(frozen=True)
class OrientationObservation(VisionObservation):
    opponent_yaw_variance: float = 0.
    opponent_age_s: float = 40.
    opponent_yaw_age_s: float = 40.
    opponent_capture_time_s: float = 0.
    opponent_position_x_variance: float = 0.
    opponent_position_y_variance: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if (any(getattr(self, name) < 0 for name in ('opponent_yaw_variance', 'opponent_age_s',
                'opponent_yaw_age_s', 'opponent_position_x_variance', 'opponent_position_y_variance')) or
                self.opponent_capture_time_s > self.decision_time_s):
            raise ValueError('inconsistent measured orientation observation')

    def to_array(self):
        return np.asarray(astuple(self), dtype=np.float64)

    @classmethod
    def from_array(cls, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(OBSERVATION_FIELDS),):
            raise ValueError('rl_obs_v3 required; v1/v2 checkpoints are not interchangeable')
        return cls(*(float(v) for v in values))


OBSERVATION_FIELDS = tuple(f.name for f in fields(OrientationObservation))
FIELD = {name: i for i, name in enumerate(OBSERVATION_FIELDS)}


def from_vision_state(state):
    if state['version'] != 'vision_state_v2':
        raise ValueError('vision_state_v2 required')
    base = from_v1(dict(state, version='vision_state_v1'))
    o = state['opponent']
    cov = o.get('position_covariance', ((0., 0.), (0., 0.)))
    return OrientationObservation(*astuple(base), opponent_yaw_variance=o['yaw_variance'],
           opponent_age_s=o['age_s'], opponent_yaw_age_s=o['yaw_age_s'],
           opponent_capture_time_s=o['yaw_capture_s'],
           opponent_position_x_variance=cov[0][0], opponent_position_y_variance=cov[1][1])
