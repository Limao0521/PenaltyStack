"""Experimental six-defense selection on the unchanged vision keeper v2.

Only the name of a central motor defense changes. Ball prediction, incoming
confirmation, side classification and commitment timing are delegated to the
published v2 implementation. No opponent yaw, target or scene truth is read.
The wide guards are explicit comparison variants, not tuned crossing limits.
"""
from penalty.keeper_defenses import KeeperDefenseAction, ACTION_VERSION, DEFENSE_NAMES
from penalty.keeper_vision import POLICY_VERSION as CORE_POLICY_VERSION, VisionKeeper


POLICY_VERSION = 'vision_keeper_v6'
STATE_VERSION = 'vision_state_v2'
MODES = ('original', 'center_right', 'wide_left', 'wide_right', 'balanced')


class VisionLibraryKeeper:
    """Select a fixed B-Human defense without changing v2's shot decision.

    ``original`` is the compatibility control. ``balanced`` mirrors the
    central guard for negative crossing y; positive y is the keeper's left.
    The remaining modes replace every central guard with the named defense.
    None of these experimental choices claims improved save performance.
    """

    def __init__(self, mode='original', **settings):
        if mode not in MODES:
            raise ValueError('unknown keeper defense mode: %s' % mode)
        self.mode = mode
        self.ball_keeper = VisionKeeper(**settings)
        self.reset()

    def reset(self):
        self.ball_keeper.reset()
        self.committed_defense = None
        self.decision = self._describe_decision(self.ball_keeper.decision, 'center')

    def describe(self):
        core = self.ball_keeper
        settings = {name: getattr(core, name) for name in (
            'corner_y', 'dive_lead_s', 'min_incoming_speed', 'confirm_frames',
            'history', 'max_ball_age', 'sigma_margin', 'goal_line_x')}
        return {'policy': POLICY_VERSION, 'core_policy': CORE_POLICY_VERSION,
                'mode': self.mode, 'settings': settings,
                'action_version': ACTION_VERSION, 'defenses': list(DEFENSE_NAMES),
                'inputs': 'vision_state_v2 ball measurements only',
                'orientation_used': False, 'experimental': True}

    def _describe_decision(self, core_decision, defense):
        return dict(core_decision, policy=POLICY_VERSION, core_policy=CORE_POLICY_VERSION,
                    mode=self.mode, selected_defense=defense, pose_prior_used=False)

    def _select(self, side, crossing):
        if side:
            return 'left' if side > 0 else 'right'
        if self.mode == 'balanced':
            return 'center_right' if crossing is not None and crossing < 0 else 'center'
        return 'center' if self.mode == 'original' else self.mode

    def __call__(self, state):
        # Copy only fields consumed by v2. The orientation and ideal evaluation
        # fields may be absent or inaccessible and cannot alter the action.
        try:
            version = state.get('version')
            core_state = {'version': 'vision_state_v1' if version == STATE_VERSION else None}
            for name in ('time_s', 'capture_s', 'remaining_time', 'ball'):
                if name in state:
                    core_state[name] = state[name]
            action = self.ball_keeper(core_state)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            action = self.ball_keeper.wait('invalid_observation')

        # A backwards clock resets the core. A normal deadline merely disables
        # its action, preserving the previously selected defense until reset.
        if self.ball_keeper.committed is None:
            self.committed_defense = None
        defense = self._select(action['block_side'], self.ball_keeper.decision.get('crossing_y'))
        if action['trigger'] >= .5:
            if self.committed_defense is None:
                self.committed_defense = defense
            defense = self.committed_defense
        self.decision = self._describe_decision(self.ball_keeper.decision, defense)
        self.decision['core_side'] = action['block_side']
        return KeeperDefenseAction(defense, action['trigger'])
