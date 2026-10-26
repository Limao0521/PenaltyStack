"""Names and units shared with the unmodified B-Human Joints enum."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '2360ab1100ad28f70e190c893f6b095651ff0bbc'
CYCLE_S = 0.012
JOINT_NAMES = (
    'headYaw', 'headPitch',
    'lShoulderPitch', 'lShoulderRoll', 'lElbowYaw', 'lElbowRoll', 'lWristYaw', 'lHand',
    'rShoulderPitch', 'rShoulderRoll', 'rElbowYaw', 'rElbowRoll', 'rWristYaw', 'rHand',
    'lHipYawPitch', 'lHipRoll', 'lHipPitch', 'lKneePitch', 'lAnklePitch', 'lAnkleRoll',
    'rHipYawPitch', 'rHipRoll', 'rHipPitch', 'rKneePitch', 'rAnklePitch', 'rAnkleRoll',
)
MODEL_JOINT_NAMES = tuple(name[0].upper() + name[1:] for name in JOINT_NAMES)
SENSOR_OFF = 10000.0
SENSOR_IGNORE = 20000.0


def executable_path():
    """An explicit build artifact; no PATH executable or silent fallback."""
    build = ROOT / 'native' / 'bhuman_motion' / 'build'
    if os.name == 'nt':
        return build / 'Release' / 'bhuman_motion.exe'
    return build / 'bhuman_motion'
