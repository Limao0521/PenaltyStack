"""Parameterized keeper block.

The original keeper keyframes (keeperJumpLeft, mirrored for the right, and
the central guard) have no tunable parameters yet; every field is bounded
and versioned but must remain nominal for the B-Human runtime.
"""
from dataclasses import dataclass

from penalty.rl.contracts import BLOCK_SIDES
from penalty.skills.base import ParameterizedSkill


@dataclass(frozen=True)
class BlockParameters:
    lateral_amplitude: float = 1.   # scale of lateral reach
    torso_lean: float = 0.          # rad
    knee_flexion: float = 0.        # rad added to nominal
    arm_extension: float = 1.       # scale
    activation_speed: float = 1.    # keyframe time scale


class ParameterizedBlock(ParameterizedSkill):
    parameters = BlockParameters
    BOUNDS = {'lateral_amplitude': (.6, 1.2), 'torso_lean': (-.2, .2), 'knee_flexion': (-.2, .2),
              'arm_extension': (.6, 1.2), 'activation_speed': (.8, 1.25)}
    RUNTIME_SUPPORTED = ()

    def __init__(self, block_side=0):
        if block_side not in BLOCK_SIDES:
            raise ValueError('lado de bloqueo desconocido')
        self.block_side = block_side
        super().__init__()

    def request(self):
        return {'block_side': BLOCK_SIDES[self.block_side]}

    def record(self):
        value = super().record()
        value.update(block_side=self.block_side, request=self.request())
        return value
