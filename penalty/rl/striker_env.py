"""PenaltyStrikerEnv: the policy chooses aim, foot, power and shot timing.

Action Box(-1, 1, 4) decodes through StrikerAction.from_normalized.
Terminations: goal, blocked, wide, out, short, fall; timeout truncates.
"""
from penalty.rl.contracts import StrikerAction
from penalty.rl.envs import PenaltyEnv


class PenaltyStrikerEnv(PenaltyEnv):
    role = 'striker'
    action_type = StrikerAction
    action_size = 4
    default_reward = 'striker_reward_v1'
    default_opponent = 'scripted'
