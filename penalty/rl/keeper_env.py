"""PenaltyKeeperEnv: the policy waits or commits a guard side once.

Action Box(-1, 1, 3) decodes through KeeperAction.from_normalized.
Terminations: save, goal, out, keeper_fall; timeout truncates. Wrap with
ObservationHistory(length=6) for the recommended keeper input.
"""
from penalty.rl.contracts import KeeperAction
from penalty.rl.envs import PenaltyEnv


class PenaltyKeeperEnv(PenaltyEnv):
    role = 'keeper'
    action_type = KeeperAction
    action_size = 3
    default_reward = 'keeper_reward_v1'

    def __init__(self, backend='synthetic', opponent=None, reward_id=None, backend_options=None):
        if opponent is None:
            opponent = 'synthetic' if backend == 'synthetic' else 'scripted'
        super().__init__(backend, opponent, reward_id, backend_options)
