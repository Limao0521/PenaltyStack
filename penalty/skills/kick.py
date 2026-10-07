"""Parameterized B-Human forwardFast kick.

The original KickEngine currently exposes the kick type (foot) and requested
reach (`kick_length_mm`, 1500..3500). `power` maps linearly onto that reach;
the remaining kinematic parameters are declared with bounds so experiments
are versioned, but stay at nominal until a generated native adaptation under
native/bhuman_motion/build/generated exposes them.
"""
from dataclasses import dataclass

from penalty.rl.contracts import KICK_TYPES
from penalty.skills.base import ParameterizedSkill

# The session accepts 1500..3500 mm, but in MuJoCo a 3100 mm request drives
# a striker joint past the runtime's 100 rad/s sensor bound (2026-10-07 probe,
# both feet, static keeper). 1500..2900 mm completed cleanly on both feet.
KICK_LENGTH_MM = (1500., 2900.)
NOMINAL_LENGTH_MM = 2500.
NOMINAL_POWER = (NOMINAL_LENGTH_MM - KICK_LENGTH_MM[0])/(KICK_LENGTH_MM[1] - KICK_LENGTH_MM[0])


@dataclass(frozen=True)
class KickParameters:
    power: float = NOMINAL_POWER    # 0..1 -> requested reach 1500..2900 mm
    swing_amplitude: float = 1.     # scale of the swing keyframes
    hip_rotation: float = 0.        # rad
    foot_angle: float = 0.          # rad
    contact_phase: float = .5       # fraction of the strike phase
    support_lean: float = 0.        # rad


def power_to_length_mm(power):
    low, high = KICK_LENGTH_MM
    return low + (high-low)*power


class ParameterizedKick(ParameterizedSkill):
    parameters = KickParameters
    BOUNDS = {'power': (0., 1.), 'swing_amplitude': (.7, 1.3), 'hip_rotation': (-.2, .2),
              'foot_angle': (-.25, .25), 'contact_phase': (0., 1.), 'support_lean': (-.1, .1)}
    RUNTIME_SUPPORTED = ('power',)

    def __init__(self, kick_side=0):
        if kick_side not in KICK_TYPES:
            raise ValueError('lado de patada desconocido')
        self.kick_side = kick_side
        super().__init__()

    def request(self):
        return {'kick_type': KICK_TYPES[self.kick_side],
                'kick_length_mm': power_to_length_mm(self.current.power)}

    def record(self):
        value = super().record()
        value.update(kick_side=self.kick_side, request=self.request())
        return value
