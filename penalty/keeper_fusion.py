"""Vision keepers v4/v5: pre-shot body yaw fused with the measured ball track.

v4 keeps the v2 ball logic unchanged (median of constant-velocity crossing
projections; side guard at <=1.8 s; centre after two incoming captures). A
body-yaw prior frozen at the shot onset is fused with each projection, and a
centre commit waits while that prior points to a corner and time remains.
Without a valid prior every decision equals v2.

v5 replaces the ball logic: a least-squares line through the incoming track,
time to the goal line from the fitted speed, and centre guards also wait for
the arrival lead unless the track is clearly central, so later captures
refine boundary cases. `use_pose=False` gives the ball-only v5 used to
attribute the gain of orientation.

Inputs are vision_state_v2 only (camera ball track, B-Human robot box, visual
yaw with age/variance). No target, contact event or scene pose is read.
The pose model is fitted on development seeds of the simulated camera
pipeline; it is not validated for physical NAO images.
"""
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from penalty.keeper_vision import median
from penalty.perception import finite

POLICY_V4, POLICY_V5, POLICY_V5_BALL = 'vision_keeper_v4', 'vision_keeper_v5', 'vision_keeper_v5_ball'
STATE_VERSION = 'vision_state_v2'
DEFAULT_POSE_MODEL = Path(__file__).resolve().parents[1]/'models'/'vision'/'kick_heading_v1.json'
# Development probe (seeds 69001, |crossing| 0.18-0.36 m): late centre guard
# vs dive per side; see docs/PORTERO_FUSION.md. Frozen before the final test.
V5_LIMITS = {'corner_y_left': .35, 'corner_y_right': .30}


@dataclass(frozen=True)
class FusionSettings:
    corner_y: float = .3            # m, centre guard inside +-corner_y
    # Optional per-side limits (keeper frame, +y = left). The original centre
    # guard (squatArmsBackLeft) is asymmetric; None keeps the symmetric corner_y.
    corner_y_left: float = None
    corner_y_right: float = None
    dive_lead_s: float = 1.8        # side guard when the arrival is this close
    centre_lead_s: float = 1.8      # v5: an uncertain centre waits for this lead
    centre_confidence: float = None # v5: if set, |y|+k*sigma inside the limits commits the centre at once
    min_incoming_speed: float = .12
    confirm_frames: int = 2
    history: int = 3
    max_ball_age: float = .25
    sigma_margin: float = 1.5
    goal_line_x: float = -.12
    # Track fit (v5).
    fit_points: int = 10            # most recent incoming captures in the fit
    min_fit_points: int = 3
    max_track_gap_s: float = .35    # longer loss restarts the track
    track_sigma_floor: float = .02  # m, model error of a straight rolling track
    # Pose prior.
    pose_window_s: float = 2.5      # pre-shot samples kept before the onset
    pose_samples: int = 5           # most recent samples used for the prior
    min_pose_samples: int = 2
    max_yaw_sigma: float = math.radians(15)
    hold_margin: float = .05        # prior must clear corner_y by this much to hold a centre
    hold_sigmas: float = 2.         # ...and disagree with the ball by this many combined std
    rest_tolerance: float = .06     # m: the ball leaving its rest point freezes the pose prior
    unfreeze_s: float = 1.          # a nudge that stops (<3 cm in this time, no shot) unfreezes

    def __post_init__(self):
        for name, low, high in (('corner_y_left', .1, .6), ('corner_y_right', .1, .6), ('centre_confidence', 0., 5.)):
            value = getattr(self, name)
            if value is not None and (not finite(value) or not low <= value <= high):
                raise ValueError('%s fuera de [%g, %g]' % (name, low, high))
        for name, low, high in (('corner_y', .15, .6), ('dive_lead_s', .3, 3.), ('centre_lead_s', .3, 3.),
                                ('min_incoming_speed', .05, .5), ('max_ball_age', .01, .5),
                                ('sigma_margin', 0., 4.), ('goal_line_x', -.5, 0.),
                                ('max_track_gap_s', .1, 1.), ('track_sigma_floor', 0., .2),
                                ('pose_window_s', .2, 6.), ('max_yaw_sigma', 0., 1.), ('hold_margin', 0., .5),
                                ('hold_sigmas', 0., 6.), ('rest_tolerance', .02, .3), ('unfreeze_s', .2, 5.)):
            value = getattr(self, name)
            if not finite(value) or not low <= value <= high:
                raise ValueError('%s fuera de [%g, %g]' % (name, low, high))
        for name, low, high in (('confirm_frames', 1, 6), ('history', 1, 6), ('fit_points', 2, 30),
                                ('min_fit_points', 2, 30), ('pose_samples', 1, 20), ('min_pose_samples', 1, 20)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('%s debe ser entero en [%d, %d]' % (name, low, high))


@dataclass(frozen=True)
class PoseModel:
    """Crossing from the striker's heading line through the ball.

    pointing_y is where the heading line through the ball meets the goal
    line. crossing ~= intercept + slope*pointing_y, residual std `sigma`.
    `ball_sigma[i]` is the std of the v2 median projection after i+1
    incoming captures (last value repeats).
    """
    intercept: float = 0.
    slope: float = 1.
    sigma: float = .12
    ball_sigma: tuple = (.25, .15, .10, .08)
    version: str = 'kick_heading_v1'
    provenance: str = 'untrained default'

    @classmethod
    def load(cls, path=DEFAULT_POSE_MODEL):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != 'kick_heading_v1':
            raise ValueError('modelo de orientación incompatible: %s' % path)
        model = cls(float(data['intercept']), float(data['slope']), float(data['sigma']),
                    tuple(float(v) for v in data['ball_sigma']), data['version'], data.get('provenance', ''))
        if (not all(finite(v) for v in (model.intercept, model.slope, model.sigma)) or model.sigma <= 0 or
                not model.ball_sigma or not all(finite(v) and v > 0 for v in model.ball_sigma)):
            raise ValueError('modelo de orientación inválido')
        return model

    def ball_projection_sigma(self, count):
        return self.ball_sigma[min(max(count, 1), len(self.ball_sigma))-1]


def pointing_y(ball, yaw, goal_line_x):
    """y where the heading line through the ball meets x=goal_line_x, or None."""
    hx, hy = math.cos(yaw), math.sin(yaw)
    if hx > -.2:  # not facing the goal
        return None
    return ball[1]+hy*(goal_line_x-ball[0])/hx


def fit_track(points, goal_line_x, now, sigma_floor):
    """Least squares y(x) and x(t) through (t, x, y); crossing, time to line, std."""
    n = len(points)
    ts, xs, ys = [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]
    mt, mx, my = sum(ts)/n, sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    stt = sum((t-mt)**2 for t in ts)
    if sxx < 1e-6 or stt < 1e-6:
        return None
    slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx
    speed = sum((t-mt)*(x-mx) for t, x in zip(ts, xs))/stt
    if not speed < -1e-3:
        return None
    crossing = my+slope*(goal_line_x-mx)
    x_now = mx+speed*(now-mt)
    arrival = (goal_line_x-x_now)/speed
    residual = sum((y-my-slope*(x-mx))**2 for x, y in zip(xs, ys))
    s2 = residual/(n-2) if n > 2 else 0.
    se = math.sqrt(max(0., s2*(1./n+(goal_line_x-mx)**2/sxx)))
    return {'crossing_y': crossing, 'time_to_line': arrival, 'speed': speed,
            'sigma': math.hypot(se, sigma_floor), 'points': n}


class FusionKeeper:
    def __init__(self, ball_logic='v2', use_pose=True, settings=None, pose_model=None):
        if ball_logic not in ('v2', 'v5'):
            raise ValueError('ball_logic debe ser v2 o v5')
        self.ball_logic, self.use_pose = ball_logic, bool(use_pose)
        self.settings = settings or FusionSettings()
        if not isinstance(self.settings, FusionSettings):
            self.settings = FusionSettings(**self.settings)
        self.pose_model = pose_model or (PoseModel.load() if DEFAULT_POSE_MODEL.exists() else PoseModel())
        self.policy = (POLICY_V4 if ball_logic == 'v2' else POLICY_V5 if use_pose else POLICY_V5_BALL)
        if ball_logic == 'v2' and not use_pose:
            raise ValueError('v2 sin orientación es vision_keeper_v2')
        self.reset()

    @classmethod
    def v4(cls, **settings):
        return cls('v2', True, FusionSettings(**settings) if settings else None)

    @classmethod
    def v5(cls, use_pose=True, **settings):
        """Published v5 limits: the left-biased original centre guard covers
        the keeper's left to 0.35 m; the right uses the dive beyond 0.30 m."""
        return cls('v5', use_pose, FusionSettings(**dict(V5_LIMITS, **settings)))

    def describe(self):
        return {'policy': self.policy, 'ball_logic': self.ball_logic, 'use_pose': self.use_pose,
                'settings': asdict(self.settings), 'pose_model': asdict(self.pose_model),
                'inputs': 'vision_state_v2 only; simulated camera appearance domain'}

    def reset(self):
        self.last_capture = self.last_time = self.committed = None
        self.incoming_frames = 0
        self.trajectories = deque(maxlen=self.settings.history)
        self.track, self.track_last = [], None
        self.pose_samples = deque(maxlen=64)
        self.prior = None
        self.onset = None
        self.rest = self.freeze = self.freeze_point = None
        self.holds = 0
        self.decision = {'reason': 'waiting_for_ball', 'pose_used': False, 'policy': self.policy}

    # -- helpers -----------------------------------------------------------
    def _wait(self, reason):
        # As in v2, an interrupted approach restarts confirmation; the onset,
        # its frozen prior and the fitted track restart with it.
        self.incoming_frames = 0
        self.trajectories.clear()
        self.track, self.track_last = [], None
        self.onset = self.prior = None
        self.decision = {'reason': reason, 'pose_used': False, 'policy': self.policy}
        return {'block_side': 0, 'trigger': 0., 'amplitude': .5}

    def _limits(self):
        s = self.settings
        return (s.corner_y if s.corner_y_left is None else s.corner_y_left,
                s.corner_y if s.corner_y_right is None else s.corner_y_right)

    def _side(self, y):
        left, right = self._limits()
        return 1 if y > left else -1 if y < -right else 0

    def _sample_pose(self, state):
        """Pre-shot body heading; every validity/age/variance gate must pass."""
        s, ball, o = self.settings, state['ball'], state['opponent']
        if not (self.use_pose and ball['position_valid'] and ball['visible'] and
                0 <= ball['age_s'] <= s.max_ball_age and o.get('position_valid') and o.get('yaw_valid') and
                0 <= o.get('age_s', 40.) <= .25 and 0 <= o.get('yaw_age_s', 40.) <= .25 and
                0 <= o.get('yaw_variance', 1e9) <= s.max_yaw_sigma**2):
            return
        stamp = o.get('yaw_capture_s', state['capture_s'])
        if self.pose_samples and stamp <= self.pose_samples[-1][0]:
            return
        y = pointing_y(ball['position'], o['yaw'], s.goal_line_x)
        if y is not None and finite(y) and abs(y) < 3.:
            self.pose_samples.append((stamp, y))

    def _track_rest(self, state):
        """Pre-shot samples stop when the ball leaves its rest point (kick contact).

        Slow shots can pass the velocity gate long after contact, while the
        striker may already turn away; the displacement marks the kick.
        """
        s, ball, capture = self.settings, state['ball'], state['capture_s']
        if not (ball['position_valid'] and ball['visible'] and 0 <= ball['age_s'] <= s.max_ball_age):
            return
        x, y = ball['position']
        if self.freeze is None:
            if self.rest is None:
                self.rest = (x, y)
            elif math.hypot(x-self.rest[0], y-self.rest[1]) > s.rest_tolerance:
                self.freeze, self.freeze_point = capture, (x, y)
            else:
                self.rest = (.9*self.rest[0]+.1*x, .9*self.rest[1]+.1*y)
        elif (self.onset is None and capture-self.freeze >= s.unfreeze_s and
              math.hypot(x-self.freeze_point[0], y-self.freeze_point[1]) < .03):
            self.freeze, self.rest = None, (x, y)

    def _freeze_prior(self, onset):
        s, m = self.settings, self.pose_model
        recent = [y for stamp, y in self.pose_samples if 0 <= onset-stamp <= s.pose_window_s][-s.pose_samples:]
        if len(recent) < s.min_pose_samples:
            return None
        heading = median(recent)
        return {'pointing_y': heading, 'y': m.intercept+m.slope*heading, 'sigma': m.sigma, 'samples': len(recent)}

    def _fuse(self, y, sigma):
        if self.prior is None:
            return y, sigma
        wb, wp = 1/sigma**2, 1/self.prior['sigma']**2
        return (wb*y+wp*self.prior['y'])/(wb+wp), math.sqrt(1/(wb+wp))

    def _pose_disagrees(self, side, y_ball, ball_sigma):
        """A centre guard is held while a fresh-at-onset corner prior contradicts the ball."""
        s, prior = self.settings, self.prior
        if side != 0 or prior is None:
            return False
        left, right = self._limits()
        corner = prior['y'] > left+s.hold_margin or prior['y'] < -right-s.hold_margin
        return corner and abs(y_ball-prior['y']) > s.hold_sigmas*math.hypot(ball_sigma, prior['sigma'])

    def _commit(self, side):
        self.committed = side
        self.decision['reason'] = 'committed'
        self.decision['side'] = side
        return {'block_side': side, 'trigger': 1., 'amplitude': .5}

    # -- policy ------------------------------------------------------------
    def __call__(self, state):
        s = self.settings
        if state.get('version') != STATE_VERSION:
            self.reset()
            return self._wait('invalid_version')
        now, capture = state['time_s'], state['capture_s']
        if not all(finite(v) for v in (now, capture, state['remaining_time'])):
            return self._wait('invalid_observation')
        if self.last_time is not None and now < self.last_time:
            self.reset()
        self.last_time = now
        if state['remaining_time'] <= 0 or now >= 28:
            return self._wait('deadline')
        if self.committed is not None:
            return {'block_side': self.committed, 'trigger': 1., 'amplitude': .5}
        if self.last_capture is not None and capture <= self.last_capture:
            return {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        self.last_capture = capture
        if self.use_pose:
            self._track_rest(state)
            if self.onset is None and self.freeze is None:
                self._sample_pose(state)
        if self.ball_logic == 'v2':
            return self._v2_ball(state)
        return self._v5_ball(state)

    def _incoming(self, ball):
        """v2 gate: fresh visible ball moving towards the keeper beyond noise."""
        s = self.settings
        if not ball['position_valid'] or not ball['visible'] or not 0 <= ball['age_s'] <= s.max_ball_age:
            return 'stale_ball'
        if not ball['velocity_valid']:
            return 'unmeasured_velocity'
        bx, by = ball['position']
        vx, vy = ball['velocity']
        variance = ball['velocity_covariance'][0][0]
        if not all(finite(v) for v in (bx, by, vx, vy, variance)) or variance < 0:
            return 'invalid_ball'
        if bx <= 0 or vx >= -(s.min_incoming_speed+s.sigma_margin*math.sqrt(variance)):
            return 'waiting_for_shot'
        return None

    def _start(self, capture):
        if self.onset is None:
            self.onset = capture
            end = self.freeze if self.freeze is not None and self.freeze <= capture else capture
            self.prior = self._freeze_prior(end) if self.use_pose else None

    def _v2_ball(self, state):
        s, ball = self.settings, state['ball']
        reason = self._incoming(ball)
        if reason is not None:
            return self._wait(reason)
        self._start(state['capture_s'])
        bx, by = ball['position']
        vx, vy = ball['velocity']
        self.incoming_frames += 1
        arrival = (s.goal_line_x-bx)/vx
        self.trajectories.append((by+vy*arrival, arrival))
        if self.incoming_frames < s.confirm_frames:
            self.decision = {'reason': 'confirming_trajectory', 'pose_used': False, 'policy': self.policy}
            return {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        y_ball = median([p[0] for p in self.trajectories])
        arrival = median([p[1] for p in self.trajectories])
        ball_sigma = self.pose_model.ball_projection_sigma(self.incoming_frames)
        y, sigma = self._fuse(y_ball, ball_sigma)
        side = self._side(y)
        self.decision = {'reason': 'waiting_for_arrival', 'crossing_y': y_ball, 'fused_y': y,
                         'fused_sigma': sigma, 'ball_sigma': ball_sigma, 'time_to_line': arrival,
                         'side': side, 'ball_side': self._side(y_ball), 'holds': self.holds,
                         'pose_used': self.prior is not None, 'policy': self.policy}
        if self.prior is not None:
            self.decision.update(pose_y=self.prior['y'], pose_sigma=self.prior['sigma'],
                                 pose_samples=self.prior['samples'])
        if side != 0 and arrival > s.dive_lead_s:
            return {'block_side': side, 'trigger': 0., 'amplitude': .5}
        # A contradicting corner prior holds a centre guard while the arrival is far.
        if arrival > s.dive_lead_s and self._pose_disagrees(side, y_ball, ball_sigma):
            self.holds += 1
            self.decision.update(reason='holding_centre', hold_reason='pose_prior_corner', holds=self.holds)
            return {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        return self._commit(side)

    def _v5_ball(self, state):
        s, ball = self.settings, state['ball']
        capture = state['capture_s']
        if self.track and capture-self.track_last > s.max_track_gap_s:
            return self._wait('track_lost')
        reason = self._incoming(ball)
        fresh = ball['position_valid'] and ball['visible'] and 0 <= ball['age_s'] <= s.max_ball_age
        if not self.track:
            if reason is not None:
                return self._wait(reason)
            self._start(capture)
        if fresh:
            x, y = ball['position']
            last = self.track[-1] if self.track else None
            if last is not None and (x > last[1]+.05 or math.hypot(x-last[1], y-last[2]) > .3):
                return self._wait('ball_left_track')
            if x > s.goal_line_x:
                self.track.append((capture, x, y))
                self.track = self.track[-s.fit_points:]
                self.track_last = capture
        if len(self.track) < max(s.min_fit_points, s.confirm_frames):
            self.decision = {'reason': 'confirming_trajectory', 'pose_used': False, 'policy': self.policy}
            return {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        fit = fit_track(self.track, s.goal_line_x, capture, s.track_sigma_floor)
        if fit is None:
            self.decision = {'reason': 'confirming_trajectory', 'pose_used': False, 'policy': self.policy}
            return {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        y, sigma = self._fuse(fit['crossing_y'], fit['sigma'])
        side = self._side(y)
        arrival = fit['time_to_line']
        self.decision = {'reason': 'waiting_for_arrival', 'crossing_y': fit['crossing_y'], 'fused_y': y,
                         'fused_sigma': sigma, 'ball_sigma': fit['sigma'], 'time_to_line': arrival,
                         'side': side, 'ball_side': self._side(fit['crossing_y']), 'track_points': fit['points'],
                         'pose_used': self.prior is not None, 'policy': self.policy}
        if self.prior is not None:
            self.decision.update(pose_y=self.prior['y'], pose_sigma=self.prior['sigma'],
                                 pose_samples=self.prior['samples'])
        # A clearly central track commits the sustained centre guard at once
        # (as v2); a centre near the boundary keeps refining until the lead.
        left, right = self._limits()
        confident_centre = (side == 0 and s.centre_confidence is not None and
                            y+s.centre_confidence*sigma <= left and y-s.centre_confidence*sigma >= -right)
        if confident_centre and self._pose_disagrees(side, fit['crossing_y'], fit['sigma']):
            confident_centre = False
            self.holds += 1
            self.decision.update(hold_reason='pose_prior_corner', holds=self.holds)
        if not confident_centre and arrival > (s.dive_lead_s if side != 0 else s.centre_lead_s):
            return {'block_side': side, 'trigger': 0., 'amplitude': .5}
        return self._commit(side)
