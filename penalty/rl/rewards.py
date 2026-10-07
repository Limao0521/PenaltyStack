"""Versioned tactical rewards. Never edit a published version: add a new id."""
from dataclasses import dataclass
from typing import Optional

# sim_error: native/physics failure, terminal and scored as a fall.
STRIKER_OUTCOMES = ('goal', 'blocked', 'wide', 'out', 'short', 'fall', 'sim_error', 'timeout')
KEEPER_OUTCOMES = ('save', 'goal', 'out', 'keeper_fall', 'sim_error', 'timeout')


@dataclass(frozen=True)
class StepEvents:
    """What happened during one tactical step. `outcome` is None until terminal."""
    outcome: Optional[str] = None
    fallen: bool = False
    contact: bool = False
    early_dive: bool = False


def striker_reward_v1(events):
    reward = -.02
    reward += {'goal': 10., 'blocked': -8., 'wide': -6., 'out': -6.,
               # Ball stopped in play without reaching the goal: a missed shot.
               'short': -6.}.get(events.outcome, 0.)
    if events.fallen:
        reward -= 10.
    return reward


def keeper_reward_v1(events):
    reward = -.02
    reward += {'save': 10., 'goal': -10.}.get(events.outcome, 0.)
    if events.outcome is not None and events.contact:
        reward += 4.
    # A dive that saves is the skill; only a useless fall is penalised.
    if events.fallen and events.outcome not in (None, 'save'):
        reward -= 5.
    if events.early_dive:
        reward -= 2.
    return reward


REWARDS = {'striker_reward_v1': striker_reward_v1, 'keeper_reward_v1': keeper_reward_v1}


def get_reward(reward_id):
    try:
        return REWARDS[reward_id]
    except KeyError:
        raise ValueError('recompensa desconocida: %s' % reward_id) from None
