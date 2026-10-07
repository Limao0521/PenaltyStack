"""Tactical decisions plugged into the shared FSM without changing it.

The striker FSM keeps SEARCH_BALL -> APPROACH_BALL -> ALIGN_TO_BALL ->
ESTIMATE_GOAL; where the scripted FSM would kick, a tactical controller
holds in SELECT_SHOT until the policy fires and the safety filter allows it.
"""
import math

from penalty.control import Command, PenaltyController, _finite

# Safety envelope for releasing a kick; the FSM alignment gates are tighter.
MAX_KICK_RANGE = .25
MAX_KICK_BEARING = .15
MAX_TILT = math.radians(12)
MAX_GYRO = .4
# Kick-pose approach inside this ball range; outside, the shared FSM walks.
KICK_POSE_RANGE = .35
KICK_POSE_DISTANCE = .19
POSE_GAIN, POSE_SPEED = 1., .05       # 1/s, m/s
TURN_GAIN, POSE_TURN = 1., .25        # 1/s, rad/s


def kick_allowed(ball_range, ball_bearing, motion_busy, tilt, gyro_norm):
    """Returns (allowed, reason)."""
    values = (ball_range, ball_bearing, tilt, gyro_norm)
    if any(v is None or not math.isfinite(v) for v in values):
        return False, 'non_finite'
    if ball_range > MAX_KICK_RANGE:
        return False, 'ball_out_of_range'
    if abs(ball_bearing) > MAX_KICK_BEARING:
        return False, 'ball_bearing'
    if motion_busy:
        return False, 'motion_busy'
    if tilt > MAX_TILT or gyro_norm > MAX_GYRO:
        return False, 'unstable'
    return True, 'ok'


class TacticalStrikerController(PenaltyController):
    """`fire`, `aim_offset` and `stability` are set by the environment."""

    def __init__(self, deadline=28., stability=None):
        super().__init__('striker', deadline)
        self.fire = False
        self.aim_offset = 0.
        self.stability = stability or (lambda: (0., 0.))
        self.updates = 0
        self.kick_released = False
        self.last_rejection = 'not_ready'

    def update(self, obs):
        self.updates += 1
        if obs.goal_bearing is not None and math.isfinite(obs.goal_bearing):
            obs.goal_bearing = math.remainder(obs.goal_bearing + self.aim_offset, 2*math.pi)
        return super().update(obs)

    def _striker(self, obs):
        near = (self.state not in ('KICK', 'RECOVER') and obs.fresh_ball()
                and obs.ball_range <= KICK_POSE_RANGE and _finite(obs.goal_bearing))
        command = self._kick_pose(obs) if near else super()._striker(obs)
        if not command.kick:
            self.last_rejection = 'not_ready'
            return command
        tilt, gyro = self.stability()
        allowed, reason = kick_allowed(obs.ball_range, obs.ball_bearing,
                                       bool(obs.motion_busy), tilt, gyro)
        if self.fire and allowed:
            self.kick_released = True
            self.last_rejection = 'released'
            return command
        self.last_rejection = reason if self.fire else 'waiting'
        self.kicked = False
        self._state('SELECT_SHOT', obs.now)
        return Command(self.state)

    def _kick_pose(self, obs):
        """Omnidirectional walk to the pose behind the ball facing the aim.

        Turning in place 0.2 m from the ball pushes it with the feet, so
        re-aiming moves on the circle around the ball instead. The ready
        gates are the shared FSM's, so a centred aim matches the scripted
        striker.
        """
        now, distance, bearing = obs.now, obs.ball_range, obs.ball_bearing
        aim = obs.goal_bearing
        if distance <= .22 and abs(bearing) <= .11 and abs(aim) <= .12:
            if self.kicked:
                self._state('RECOVER', now)
                return Command(self.state)
            self.kicked = True
            self._state('KICK', now)
            return Command(self.state, kick=True)
        bx, by = distance*math.cos(bearing), distance*math.sin(bearing)
        target_x = bx - KICK_POSE_DISTANCE*math.cos(aim)
        target_y = by - KICK_POSE_DISTANCE*math.sin(aim)
        heading = math.atan2(by, bx)*.5 + aim*.5
        self._state('ALIGN_TO_SHOT', now)
        return Command(self.state,
                       forward=_clip(target_x*POSE_GAIN, POSE_SPEED),
                       lateral=_clip(target_y*POSE_GAIN, POSE_SPEED),
                       turn=_clip(heading*TURN_GAIN, POSE_TURN),
                       head_yaw=_clip(bearing, .7))


def _clip(value, bound):
    return max(-bound, min(bound, value))


class TacticalKeeperController(PenaltyController):
    """Holds TRACK_BALL until `commit(side)`; then guards on that side."""

    def __init__(self, deadline=28.):
        super().__init__('keeper', deadline)
        self.block = None
        self.updates = 0

    def commit(self, side):
        if self.block is None:
            self.block = side

    def update(self, obs):
        self.updates += 1
        return super().update(obs)

    def _keeper(self, obs):
        head = max(-.5, min(.5, obs.ball_bearing)) if obs.fresh_ball() else 0.
        if self.block is None:
            self._state('TRACK_BALL', obs.now)
            return Command(self.state, head_yaw=head, block_now=False)
        self._state('BLOCK', obs.now)
        return Command(self.state, head_yaw=head, block=self.block, block_now=True)


class StaticKeeperController(TacticalKeeperController):
    """Opponent that only tracks the ball; it never commits a guard."""

    def commit(self, side):
        pass
