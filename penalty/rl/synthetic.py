"""Analytic penalty worlds for curriculum stage 1 and fast tests.

These are not physics and never touch MuJoCo: ball flight is a straight
roll with constant deceleration, and kick/dive outcomes come from simple
coverage models with the constants below. Every observation and episode is
labelled `backend='synthetic'`; nothing learned here may be claimed to hold
for B-Human/MuJoCo or hardware without re-evaluation on those backends.
"""
import math

from penalty.rl import geometry
from penalty.rl.contracts import (BLOCK_CENTER, BLOCK_LEFT, BLOCK_RIGHT, KICK_LEFT,
                                  KICK_RIGHT, TARGET_Y_LIMIT)
from penalty.rl.controllers import kick_allowed
from penalty.rl.observation import ObservationBuilder, WorldState
from penalty.rl.rewards import StepEvents

DT = .1
DECELERATION = .6          # m/s^2 rolling resistance
KICK_SPEED = (1.2, 4.0)    # m/s at power 0 and 1
TURN_RATE = .3             # rad/s while re-aiming
ALIGN_TOLERANCE = .12      # rad, as the FSM's goal-bearing gate
KEEPER_LINE_X = geometry.KEEPER_START[0][0]
BODY_HALF_WIDTH = .2
CENTER_GUARD = .25
SIDE_THRESHOLD = .2
BALL_TO_ROBOT = .17

STRIKER_OPPONENTS = ('empty', 'static', 'random', 'scripted')
KEEPER_OPPONENTS = ('synthetic',)
INFO = {'backend': 'synthetic', 'privileged_inputs': ['analytic ball and robot poses'],
        'synthetic_inputs': ['analytic kick, roll and dive models', 'IMU (zero)']}


def roll(origin, direction, speed, elapsed):
    """Ball position and speed after `elapsed` seconds of rolling."""
    stop = speed/DECELERATION
    t = min(max(0., elapsed), stop)
    distance = speed*t - .5*DECELERATION*t*t
    return ((origin[0] + math.cos(direction)*distance, origin[1] + math.sin(direction)*distance),
            max(0., speed - DECELERATION*t))


def line_crossing(origin, direction, speed, line_x):
    """(time, y) when the roll reaches x=line_x, or None if it stops first."""
    dx = math.cos(direction)
    if abs(dx) < 1e-9 or (line_x - origin[0])/dx <= 0:
        return None
    path = (line_x - origin[0])/dx
    disc = speed*speed - 2*DECELERATION*path
    if disc < 0:
        return None
    time = (speed - math.sqrt(disc))/DECELERATION
    return time, origin[1] + math.sin(direction)*path


def side_of(y):
    return BLOCK_LEFT if y > SIDE_THRESHOLD else BLOCK_RIGHT if y < -SIDE_THRESHOLD else BLOCK_CENTER


def covered(y, keeper_y, side, elapsed, amplitude=.5):
    """Dive/guard coverage `elapsed` seconds after the keeper commits.

    Sides are in world y (keeper faces +x, so its left is +y)."""
    rel = y - keeper_y
    if side is None or elapsed < .15:
        return abs(rel) < BODY_HALF_WIDTH
    if side == BLOCK_CENTER:
        return abs(rel) < (CENTER_GUARD if elapsed >= .3 else BODY_HALF_WIDTH)
    ramp = .25 + .2*amplitude
    full = .35 + .35*amplitude
    reach = BODY_HALF_WIDTH + (full - BODY_HALF_WIDTH)*min(1., (elapsed-.15)/ramp)
    return 0.05 <= rel*side <= reach


class _World:
    control_dt = DT

    def __init__(self):
        self.builder = ObservationBuilder()

    def close(self):
        pass


class SyntheticStrikerWorld(_World):
    def __init__(self, opponent='scripted'):
        if opponent not in STRIKER_OPPONENTS:
            raise ValueError('oponente sintético desconocido: %s' % opponent)
        self.opponent = opponent
        super().__init__()

    def reset(self, rng, options):
        self.rng = rng
        self.time = 0.
        self.ball = (geometry.PENALTY_MARK[0] + rng.uniform(-.02, .02), rng.uniform(-.03, .03))
        self.heading = math.pi + rng.uniform(-.06, .06)
        self.keeper_y = 0. if self.opponent != 'empty' else 3.
        self.keeper_vy = 0.
        self.kick = None
        self.builder.reset()
        return self._observe(), dict(INFO, opponent=self.opponent)

    def _robot_xy(self):
        return (self.ball[0] - BALL_TO_ROBOT*math.cos(self.heading),
                self.ball[1] - BALL_TO_ROBOT*math.sin(self.heading))

    def _observe(self):
        return self.builder.build(WorldState(
            time_s=self.time, remaining_time=geometry.PENALTY_SECONDS - self.time,
            ball_xy=self.ball, robot_xy=self._robot_xy(), robot_yaw=self.heading,
            opponent_xy=(KEEPER_LINE_X, self.keeper_y), opponent_yaw=0.))

    def _target_heading(self, target_y):
        # The striker's left is world -y (see geometry.LEFT_POST).
        return math.atan2(-target_y - self.ball[1], geometry.GOAL_LINE_X - self.ball[0])

    def step(self, action):
        self.time += DT
        if self.opponent == 'random' and self.kick is None:
            # Pre-shot feints the policy can observe.
            self.keeper_vy = max(-.3, min(.3, self.keeper_vy + self.rng.normal(0, .15)))
            self.keeper_y = max(-.3, min(.3, self.keeper_y + self.keeper_vy*DT))
        error = geometry.wrap(self._target_heading(action.target_y) - self.heading)
        aligned = abs(error) < ALIGN_TOLERANCE
        if not aligned:
            self.heading += max(-TURN_RATE*DT, min(TURN_RATE*DT, error))
        details = {'safety': 'waiting'}
        if action.fires:
            allowed, reason = kick_allowed(BALL_TO_ROBOT, 0., False, 0., 0.)
            if allowed and not aligned:
                allowed, reason = False, 'not_aligned'
            details['safety'] = reason
            if allowed:
                return self._shoot(action, details)
        if self.time >= geometry.PENALTY_SECONDS - 1e-9:
            return self._observe(), StepEvents('timeout'), details
        return self._observe(), StepEvents(), details

    def _shoot(self, action, details):
        rng = self.rng
        power = action.kick_power
        # A left foot crossing to the striker's left is the awkward side.
        awkward = ((action.kick_side == KICK_LEFT and action.target_y > SIDE_THRESHOLD) or
                   (action.kick_side == KICK_RIGHT and action.target_y < -SIDE_THRESHOLD))
        sigma = .04 + .06*power + (.03 if awkward else 0.)
        direction = self.heading + rng.normal(0, sigma)
        speed = KICK_SPEED[0] + (KICK_SPEED[1]-KICK_SPEED[0])*power
        fallen = rng.random() < .01 + .06*power*power
        crossing = line_crossing(self.ball, direction, speed, geometry.GOAL_LINE_X)
        keeper = line_crossing(self.ball, direction, speed, KEEPER_LINE_X)
        dive, reaction = None, 0.
        if self.opponent == 'random':
            dive, reaction = int(rng.integers(-1, 2)), .25
        elif self.opponent == 'scripted' and keeper is not None:
            dive, reaction = side_of(keeper[1] + rng.normal(0, .15)), .35
        if fallen:
            outcome = 'fall'
        elif crossing is None:
            outcome = 'short'
        elif (self.opponent != 'empty' and keeper is not None and
              covered(keeper[1], self.keeper_y, dive, keeper[0]-reaction)):
            outcome = 'blocked'
        elif abs(crossing[1]) < geometry.GOAL_HALF_WIDTH:
            outcome = 'goal'
        else:
            outcome = 'wide'
        # Report in the striker's convention (left positive = world -y).
        crossing_y = None if crossing is None else -crossing[1]
        self.kick = {'kick_side': action.kick_side, 'kick_power': power, 'target_y': action.target_y,
                     'ball_speed': speed, 'crossing_y': crossing_y, 'kick_time': self.time,
                     'target_error': None if crossing_y is None else abs(crossing_y - action.target_y),
                     'fallen': fallen, 'result': outcome}
        details['kick_record'] = self.kick
        ball, _ = roll(self.ball, direction, speed, 99.)
        self.ball = ball if crossing is None else (geometry.GOAL_LINE_X, crossing[1])
        return self._observe(), StepEvents(outcome, fallen=fallen), details

    def episode_record(self):
        kick = self.kick or {}
        return {name: kick.get(name) for name in ('ball_speed', 'target_error', 'kick_time',
                                                  'crossing_y')}


class SyntheticKeeperWorld(_World):
    """Stage-1 keeper curriculum: sampled straight shots from the mark."""

    def __init__(self, opponent='synthetic', speed_range=(1.5, 4.), target_range=(-.7, .7),
                 kick_time_range=(.5, 3.)):
        if opponent not in KEEPER_OPPONENTS:
            raise ValueError('oponente sintético desconocido: %s' % opponent)
        self.opponent = opponent
        self.speed_range, self.target_range = speed_range, target_range
        self.kick_time_range = kick_time_range
        super().__init__()

    def reset(self, rng, options):
        self.rng = rng
        self.time = 0.
        self.kick_at = rng.uniform(*options.get('kick_time_range', self.kick_time_range))
        target = rng.uniform(*options.get('target_range', self.target_range))
        self.speed = rng.uniform(*options.get('speed_range', self.speed_range))
        self.origin = geometry.PENALTY_MARK
        self.direction = math.atan2(target - self.origin[1], geometry.GOAL_LINE_X - self.origin[0])
        self.direction += rng.normal(0, .03)
        self.crossing = line_crossing(self.origin, self.direction, self.speed, KEEPER_LINE_X)
        self.commit = None
        self.outcome = None
        self.builder.reset()
        return self._observe(), dict(INFO, opponent=self.opponent)

    def _ball(self, time):
        return roll(self.origin, self.direction, self.speed, time - self.kick_at)[0] \
            if time > self.kick_at else self.origin

    def _observe(self):
        return self.builder.build(WorldState(
            time_s=self.time, remaining_time=geometry.PENALTY_SECONDS - self.time,
            ball_xy=self._ball(self.time), robot_xy=geometry.KEEPER_START[0],
            robot_yaw=geometry.KEEPER_START[1], opponent_xy=geometry.STRIKER_START[0],
            opponent_yaw=geometry.STRIKER_START[1]))

    def step(self, action):
        start = self.time
        self.time += DT
        details = {'unapplied': []}
        early = False
        if action.fires and self.commit is None:
            arrival = None if self.crossing is None else self.kick_at + self.crossing[0]
            early = start < self.kick_at or (arrival is not None and arrival - start > 1.)
            self.commit = {'side': action.block_side, 'amplitude': action.amplitude,
                           'time': start, 'early_dive': early}
            if start < self.kick_at and action.block_side != BLOCK_CENTER:
                self.outcome = 'keeper_fall'
                return self._observe(), StepEvents('keeper_fall', fallen=True, early_dive=True), details
        if self.crossing is not None and self.kick_at + self.crossing[0] <= self.time:
            y = self.crossing[1]
            commit = self.commit or {}
            elapsed = self.kick_at + self.crossing[0] - commit.get('time', 1e9)
            save = covered(y, 0., commit.get('side'), elapsed, commit.get('amplitude', .5))
            goal = line_crossing(self.origin, self.direction, self.speed, geometry.GOAL_LINE_X)
            if save:
                self.outcome = 'save'
            elif goal is not None and abs(goal[1]) < geometry.GOAL_HALF_WIDTH:
                self.outcome = 'goal'
            else:
                self.outcome = 'out'
        elif self.crossing is None and self.time > self.kick_at + self.speed/DECELERATION:
            self.outcome = 'out'
        elif self.time >= geometry.PENALTY_SECONDS - 1e-9:
            self.outcome = 'timeout'
        fallen = self.commit is not None and self.commit['side'] != BLOCK_CENTER
        return (self._observe(), StepEvents(self.outcome, fallen=fallen and self.outcome is not None,
                contact=self.outcome == 'save', early_dive=early), details)

    def episode_record(self):
        commit = self.commit or {}
        shot_side = None if self.crossing is None else side_of(self.crossing[1])
        reaction = None
        if commit and commit['time'] >= self.kick_at:
            reaction = commit['time'] - self.kick_at
        return {'reaction_time': reaction, 'early_dive': bool(commit.get('early_dive')),
                'wrong_side': bool(commit) and shot_side is not None and commit['side'] != shot_side,
                'contact': self.outcome == 'save', 'shot_y': None if self.crossing is None else self.crossing[1],
                'shot_speed': self.speed}


def make_world(role, opponent, **options):
    if role == 'striker':
        return SyntheticStrikerWorld(opponent, **options)
    return SyntheticKeeperWorld(opponent, **options)


__all__ = ['SyntheticStrikerWorld', 'SyntheticKeeperWorld', 'make_world', 'TARGET_Y_LIMIT']
