"""FSM determinista independiente del simulador y del hardware.

Coordenadas: avance positivo hacia el frente del robot; bearing positivo a
la izquierda. Distancias en metros, ángulos en radianes, tiempo en segundos.
"""
from __future__ import division

import math


def _finite(value):
    return value is not None and not math.isnan(value) and not math.isinf(value)


class Observation(object):
    def __init__(self, now, start=False, abort=False, ball_range=None,
                 ball_bearing=None, goal_bearing=None, ball_age=0.0,
                 ball_lateral_speed=None, motion_busy=None, motion_done=None):
        self.now = float(now)
        self.start = bool(start)
        self.abort = bool(abort)
        self.ball_range = ball_range
        self.ball_bearing = ball_bearing
        self.goal_bearing = goal_bearing
        self.ball_age = float(ball_age)
        self.ball_lateral_speed = ball_lateral_speed
        # None retains the NAOqi MVP's timing contract; simulators with a
        # physical motor supply an explicit completion acknowledgement.
        self.motion_busy = motion_busy
        self.motion_done = motion_done

    def fresh_ball(self):
        return (self.ball_range is not None and self.ball_bearing is not None
                and self.ball_age <= 0.25 and self.ball_range >= 0
                and _finite(self.ball_range) and _finite(self.ball_bearing))


class Command(object):
    def __init__(self, state, forward=0.0, lateral=0.0, turn=0.0,
                 head_yaw=0.0, kick=False, block='center', block_now=None):
        self.state = state
        self.forward = forward
        self.lateral = lateral
        self.turn = turn
        self.head_yaw = head_yaw
        self.kick = kick
        self.block = block
        # None keeps the simulator's scripted guard timing; True/False is an
        # explicit tactical decision (RL keeper) to start or hold the guard.
        self.block_now = block_now


class PenaltyController(object):
    def __init__(self, role, deadline=28.0):
        if role not in ('striker', 'keeper'):
            raise ValueError('role must be striker or keeper')
        self.role = role
        self.deadline = float(deadline)
        self.state = 'WAIT_FOR_START'
        self.start_time = None
        self.state_time = None
        self.kicked = False
        self.abort_latched = False
        self.last_tick = None
        self.last_forward = 0.0
        self.commanded_forward = 0.0

    def _state(self, name, now):
        if self.state != name:
            self.state = name
            self.state_time = now

    def update(self, obs):
        now = obs.now
        if self.last_tick is not None:
            self.commanded_forward += max(0.0, self.last_forward) * min(0.25, max(0.0, now - self.last_tick))
        self.last_tick = now
        self.last_forward = 0.0
        if obs.abort:
            self.abort_latched = True
        if self.abort_latched:
            self._state('STOP', now)
            return Command(self.state)
        if self.state == 'STOP':
            return Command(self.state)
        if self.start_time is None:
            if not obs.start:
                return Command(self.state)
            self.start_time = now
            self._state('TRACK_BALL' if self.role == 'keeper' else 'SEARCH_BALL', now)
        if now - self.start_time >= self.deadline:
            self._state('STOP', now)
            return Command(self.state)
        if self.role == 'striker' and self.commanded_forward >= 0.6:
            self._state('STOP', now)
            return Command(self.state)
        if self.role == 'keeper':
            return self._keeper(obs)
        action = self._striker(obs)
        self.last_forward = action.forward
        return action

    def _striker(self, obs):
        now = obs.now
        if self.state == 'KICK':
            if obs.motion_done is not None and not obs.motion_done:
                return Command(self.state)
            self._state('RECOVER', now)
            return Command(self.state)
        if self.state == 'RECOVER':
            if now - self.state_time >= 1.5:
                self._state('STOP', now)
            return Command(self.state)
        if not obs.fresh_ball():
            self._state('SEARCH_BALL', now)
            sweep = math.sin((now - self.start_time) * 2.0) * 0.6
            return Command(self.state, head_yaw=sweep)
        distance = obs.ball_range
        bearing = obs.ball_bearing
        if distance > 0.22:
            self._state('APPROACH_BALL', now)
            return Command(self.state, forward=min(0.25, max(0.03, (distance - 0.18) * 0.55)),
                           turn=max(-0.35, min(0.35, bearing * 1.4)),
                           head_yaw=max(-0.7, min(0.7, bearing)))
        if abs(bearing) > 0.11:
            self._state('ALIGN_TO_BALL', now)
            return Command(self.state, turn=max(-0.22, min(0.22, bearing)))
        self._state('ESTIMATE_GOAL', now)
        if not _finite(obs.goal_bearing):
            return Command(self.state)
        if abs(obs.goal_bearing) > 0.12:
            return Command(self.state, turn=max(-0.16, min(0.16, obs.goal_bearing)))
        if self.kicked:
            self._state('RECOVER', now)
            return Command(self.state)
        self.kicked = True
        self._state('KICK', now)
        return Command(self.state, kick=True)

    def _keeper(self, obs):
        now = obs.now
        if not obs.fresh_ball():
            self._state('TRACK_BALL', now)
            return Command(self.state, head_yaw=math.sin((now - self.start_time) * 1.5) * 0.4)
        self._state('BLOCK', now)
        # La postura es estática en el MVP; la etiqueta informa el lado observado.
        side = 'center'
        if obs.ball_bearing > 0.18:
            side = 'left'
        elif obs.ball_bearing < -0.18:
            side = 'right'
        return Command(self.state, head_yaw=max(-0.5, min(0.5, obs.ball_bearing)), block=side)
