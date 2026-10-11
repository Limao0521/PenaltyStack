"""Versioned selection of original B-Human keeper primitives.

keeper_action_v1 remains unchanged. This action selects a named defense;
it never requests joint positions, reach scaling or invented trajectories.
Preparation is available to manual motion trials, outside the six defenses.
"""
from dataclasses import asdict, dataclass
import math

DEFENSE_CATALOG_VERSION = 'keeper_defenses_v2'
ACTION_VERSION = 'keeper_action_v2'
DEFENSE_NAMES = ('center', 'center_right', 'wide_left', 'wide_right', 'left', 'right')
PREPARATION_NAME = 'prepare'
BLOCK_REQUEST_NAMES = DEFENSE_NAMES + (PREPARATION_NAME,)

# Keeper coordinates: +y is the robot's left. These signs describe a
# requested side, not a promise of the movement's physical coverage.
DEFENSE_SIDES = {'center': 0, 'center_right': 0, 'wide_left': 1,
                 'wide_right': -1, 'left': 1, 'right': -1}
DEFENSE_MOTIONS = {
    'center': ('squatArmsBackLeft', 'genuflectStand', False),
    'center_right': ('squatArmsBackRight', 'genuflectStand', True),
    'wide_left': ('squatWideArmsBackLeft', 'genuflectStandDefender', False),
    'wide_right': ('squatWideArmsBackRight', 'genuflectStandDefender', True),
    'left': ('jumpLeft', 'keeperJumpLeft', False),
    'right': ('jumpRight', 'keeperJumpLeft', True),
    'prepare': ('prepare', 'sitDownKeeper', False),
}


@dataclass(frozen=True)
class KeeperDefenseAction:
    defense: str
    trigger: float

    def __post_init__(self):
        if not isinstance(self.defense, str) or self.defense not in DEFENSE_NAMES:
            raise ValueError('defensa desconocida: %s' % self.defense)
        if (isinstance(self.trigger, bool) or not isinstance(self.trigger, (int, float)) or
                not math.isfinite(self.trigger) or not 0. <= self.trigger <= 1.):
            raise ValueError('trigger debe ser un número finito en [0, 1]')

    @property
    def fires(self):
        return self.trigger >= .5

    @property
    def side(self):
        return DEFENSE_SIDES[self.defense]

    def to_dict(self):
        return dict(asdict(self), version=ACTION_VERSION)


def catalog():
    return {'version': DEFENSE_CATALOG_VERSION,
            'defenses': {name: {'request': DEFENSE_MOTIONS[name][0],
                                'keyframe': DEFENSE_MOTIONS[name][1],
                                'mirror': DEFENSE_MOTIONS[name][2]}
                         for name in DEFENSE_NAMES},
            'preparations': {PREPARATION_NAME: {'request': 'prepare',
                                              'keyframe': 'sitDownKeeper', 'mirror': False}},
            'learned_joint_trajectories': False, 'amplitude_scaling': False}
