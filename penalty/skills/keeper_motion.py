"""Versioned, bounded keeper keyframe adaptation executed by B-Human.

These parameters change real interpolation durations and joint targets in
the original KeyframeMotionEngine. They are not selector scores, root
trajectories, servo overrides or a physical NAO calibration.
"""
from dataclasses import asdict, dataclass, fields
import math
from numbers import Real

from penalty.skills.base import ParameterizedSkill

MOTION_SKILL_VERSION = 'keeper_motion_skill_v1'
MOTION_PARAMETERS_VERSION = 'keeper_motion_parameters_v1'


@dataclass(frozen=True)
class MotionParameters:
    guard_time_scale: float = 1.
    jump_time_scale: float = 1.
    arm_spread_rad: float = 0.
    guard_knee_offset_rad: float = 0.


KeeperMotionParameters = MotionParameters


class ParameterizedKeeperMotion(ParameterizedSkill):
    parameters = MotionParameters
    BOUNDS = {'guard_time_scale': (.85, 1.15), 'jump_time_scale': (.85, 1.15),
              'arm_spread_rad': (-.08, .12), 'guard_knee_offset_rad': (0., .10)}
    RUNTIME_SUPPORTED = tuple(BOUNDS)

    @classmethod
    def validate(cls, params):
        if not isinstance(params, MotionParameters):
            raise TypeError('se esperaba MotionParameters')
        for field in fields(params):
            value = getattr(params, field.name)
            low, high = cls.BOUNDS[field.name]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError('%s debe ser un número finito' % field.name)
            if not low <= value <= high:
                raise ValueError('%s fuera de [%g, %g]' % (field.name, low, high))
        return params

    def request(self):
        return {'keeper_motion_skill': {'version': MOTION_SKILL_VERSION,
                                        'parameters': asdict(self.current)}}

    def record(self):
        return dict(super().record(), version=MOTION_SKILL_VERSION,
                    parameters_version=MOTION_PARAMETERS_VERSION,
                    controller='original B-Human KeyframeMotionEngine',
                    learned_scope='keeper keyframe durations, shoulder roll and guard knee pitch',
                    executed_on_hardware=False)


def motion_parameters(value):
    """Validate a dataclass or an exact parameter mapping; None means legacy."""
    if value is None:
        return None
    if isinstance(value, dict):
        names = set(ParameterizedKeeperMotion.RUNTIME_SUPPORTED)
        if set(value) != names:
            raise ValueError('el perfil debe contener exactamente los cuatro parámetros')
        value = MotionParameters(**value)
    return ParameterizedKeeperMotion.validate(value)
