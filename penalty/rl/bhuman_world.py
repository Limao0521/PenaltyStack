"""Tactical decisions over the original B-Human engine in MuJoCo.

One tactical step spans one 0.1 s FSM control period of SimulationSession.
Locomotion, kick and guard execution stay with the shared FSM and B-Human;
the policy only replaces the shot decision (striker) or the guard decision
(keeper). Ball and robot poses come from the scene (ideal perception) and
are declared as privileged in every episode's info.
"""
import math

import numpy as np

from penalty.control import PenaltyController
from penalty.rl import geometry
from penalty.rl.contracts import BLOCK_SIDES, KICK_LEFT, KICK_RIGHT, TARGET_Y_LIMIT, StrikerAction
from penalty.rl.controllers import (StaticKeeperController, TacticalKeeperController,
                                    TacticalStrikerController)
from penalty.rl.observation import ObservationBuilder, WorldState, torso_roll_pitch
from penalty.rl.rewards import StepEvents
from penalty.rl.synthetic import side_of
from penalty.skills.kick import KickParameters, ParameterizedKick

STRIKER_OPPONENTS = ('scripted', 'static')
KEEPER_OPPONENTS = ('scripted', 'random_kicks')
# After a kick or committed guard the episode runs to its ball outcome.
FAST_FORWARD_S = 10.
STOPPED_SPEED = .05
STOPPED_FOR_S = .5
CENTER_SHOT_Y = .15
MAX_BALL_JITTER = .2
MAX_START_JITTER = (.2, .5)
INFO = {'backend': 'bhuman',
        'privileged_inputs': ['scene ball position', 'scene robot/opponent poses',
                              'torso orientation'],
        'synthetic_inputs': ['hand positions', 'motor currents/temperatures', 'joint wear']}


def sample_shot(rng, target=None, center=0., power=(.25, 1.)):
    """random_kicks opponent shot. target=None keeps the original uniform aim
    over the whole goal; otherwise |aim| is drawn from `target` on a random
    side, except a `center` fraction aimed within ±CENTER_SHOT_Y."""
    if target is None:
        aim = float(rng.uniform(-TARGET_Y_LIMIT, TARGET_Y_LIMIT))
    elif rng.random() < center:
        aim = float(rng.uniform(-CENTER_SHOT_Y, CENTER_SHOT_Y))
    else:
        aim = float(rng.choice((-1., 1.))*rng.uniform(*target))
    return StrikerAction(aim, int(rng.integers(KICK_LEFT, KICK_RIGHT+1)), float(rng.uniform(*power)), 1.)


def _check_range(value, low, high, label):
    if (len(value) != 2 or not all(math.isfinite(v) for v in value)
            or not low <= value[0] <= value[1] <= high):
        raise ValueError('%s debe ser (min, max) dentro de [%g, %g]' % (label, low, high))
    return tuple(float(v) for v in value)


class BHumanWorld:
    control_dt = .1

    def __init__(self, role, opponent='scripted', seed=0, record_dir=None, simulator_factory=None,
                 kick_parameters=None, ball_jitter=0., start_jitter=(0., 0.), shot_target=None,
                 center_shots=0., shot_power=(.25, 1.)):
        """ball_jitter: ± metres around the penalty mark per episode.
        start_jitter: (± metres, ± rad) per robot from the nominal start pose.
        shot_target/center_shots/shot_power: random_kicks distribution (sample_shot)."""
        allowed = STRIKER_OPPONENTS if role == 'striker' else KEEPER_OPPONENTS
        if role not in ('striker', 'keeper') or opponent not in allowed:
            raise ValueError('combinación rol/oponente no soportada por B-Human: %s/%s' % (role, opponent))
        self.ball_jitter = float(ball_jitter)
        if not 0 <= self.ball_jitter <= MAX_BALL_JITTER:
            raise ValueError('ball_jitter debe estar en [0, %g] m' % MAX_BALL_JITTER)
        self.start_jitter = tuple(float(v) for v in start_jitter)
        if (len(self.start_jitter) != 2 or not all(0 <= v <= limit for v, limit in
                                                    zip(self.start_jitter, MAX_START_JITTER))):
            raise ValueError('start_jitter debe ser (m <= %g, rad <= %g)' % MAX_START_JITTER)
        self.shot_target = (None if shot_target is None else
                            _check_range(shot_target, 0., TARGET_Y_LIMIT, 'shot_target'))
        self.center_shots = float(center_shots)
        if not 0 <= self.center_shots <= 1:
            raise ValueError('center_shots debe estar en [0, 1]')
        self.shot_power = _check_range(shot_power, 0., 1., 'shot_power')
        self.role, self.opponent = role, opponent
        self.other = 'keeper' if role == 'striker' else 'striker'
        self.base_seed = seed
        self.record_dir = record_dir
        self.simulator_factory = simulator_factory
        self.kick_parameters = kick_parameters or KickParameters()
        ParameterizedKick().validate(self.kick_parameters)
        self.session = None
        self.builder = ObservationBuilder()

    # -- lifecycle -------------------------------------------------------
    def _ensure_session(self):
        if self.session is None:
            from penalty.sim_session import SimulationSession
            if self.simulator_factory is None:
                from penalty.physical_sim import PhysicalSimulator
                sim = PhysicalSimulator(seed=self.base_seed)
            else:
                sim = self.simulator_factory()
            self.session = SimulationSession(sim, record_dir=self.record_dir, mode='penalty',
                                             publish_ticks=False)
        return self.session

    @property
    def sim(self):
        return self.session.sim

    def close(self):
        session, self.session = self.session, None
        if session is not None:
            session.close()
            try:
                session.tick()
            except Exception:
                # The session is being discarded (often after a native
                # failure); a failing final tick must not skip collection.
                pass
            session.sim.close()

    def _layout(self, rng, options):
        """Ball position and robot start offsets; no draws without jitter."""
        if 'ball_xy' in options:
            ball_xy = options['ball_xy']
        elif self.ball_jitter:
            jitter = self.ball_jitter
            ball_xy = (geometry.PENALTY_MARK[0] + rng.uniform(-jitter, jitter),
                       geometry.PENALTY_MARK[1] + rng.uniform(-jitter, jitter))
        else:
            ball_xy = geometry.PENALTY_MARK
        offsets = None
        if any(self.start_jitter):
            distance, yaw = self.start_jitter
            offsets = {role: (float(rng.uniform(-distance, distance)), float(rng.uniform(-distance, distance)),
                              float(rng.uniform(-yaw, yaw))) for role in ('striker', 'keeper')}
        return ball_xy, offsets

    def reset(self, rng, options):
        ball_xy, offsets = self._layout(rng, options)
        ball_xy = tuple(float(v) for v in ball_xy)
        if len(ball_xy) != 2 or not all(math.isfinite(v) and abs(v) <= 1.2 for v in ball_xy):
            raise ValueError('ball_xy debe contener dos coordenadas finitas dentro de la cancha')
        session = self._ensure_session()
        sim = session.sim
        sim.seed = int(options.get('engine_seed', rng.integers(0, 2**31-1)))
        sim.ball_xy = ball_xy
        if offsets is not None:
            sim.start_offsets = offsets
        self.builder.reset()
        self.kick_time = self.contact_time = self.commit_time = None
        self.max_ball_speed = 0.
        self.stopped_for = 0.
        self.outcome = None
        self.commit_side = None
        self.early_dive = False
        self.kick_request = None
        try:
            session.command('reset')
            self._install_controllers(rng, options)
            session.command('start')
            self._advance()
            if session.state != 'RUNNING':
                raise RuntimeError('B-Human no inició el penal: %s' % session.error)
        except Exception:
            # Never reuse a half-started session: collect both engines.
            self.close()
            raise
        info = dict(INFO, opponent=self.opponent, engine_seed=sim.seed, trial=session.trial,
                    ball_xy=list(ball_xy), start_offsets=offsets)
        if self.role == 'keeper' and self.opponent == 'random_kicks':
            info['opponent_shot'] = dict(self.opponent_shot)
        return self._observe(), info

    def _install_controllers(self, rng, options):
        session = self.session
        if self.role == 'striker':
            self.striker = TacticalStrikerController(stability=lambda: self._stability('striker'))
            session.controllers['striker'] = self.striker
            if self.opponent == 'static':
                session.controllers['keeper'] = StaticKeeperController()
            return
        self.keeper = TacticalKeeperController()
        session.controllers['keeper'] = self.keeper
        if self.opponent == 'random_kicks':
            # Parameterized B-Human kicks: random aim, foot and reach.
            shot = sample_shot(rng, self.shot_target, self.center_shots, self.shot_power)
            shot = options.get('opponent_shot', shot)
            self.opponent_shot = {'target_y': shot.target_y, 'kick_side': shot.kick_side,
                                  'kick_power': shot.kick_power}
            striker = TacticalStrikerController(stability=lambda: self._stability('striker'))
            striker.fire = True
            session.controllers['striker'] = striker
            self._opponent_striker, self._opponent_target = striker, shot.target_y
            kick = ParameterizedKick(shot.kick_side)
            session.manual.update(kick.apply(KickParameters(power=shot.kick_power)))
        else:
            self._opponent_striker = None
            session.controllers['striker'] = PenaltyController('striker')
            session.manual.update(kick_type='forwardFastLeft', kick_length_mm=2500.)

    # -- state extraction ------------------------------------------------
    def _ground_pose(self, role):
        sim = self.sim
        feet = sim.adapter.feet[role]
        xy = (sim.data.xpos[feet[0], :2] + sim.data.xpos[feet[1], :2]) * .5
        return (float(xy[0]), float(xy[1])), sim.pose(role)[1]

    def _stability(self, role):
        sim = self.sim
        body = sim.model.body(role + '_Nao').id
        tilt = math.acos(float(np.clip(sim.data.xmat[body].reshape(3, 3)[2, 2], -1, 1)))
        gyro = float(np.linalg.norm(sim.data.sensor(role + '_imu_gyro').data))
        return tilt, gyro

    def _ball_xy(self):
        q = self.sim.data.qpos[self.sim.ball_qpos:self.sim.ball_qpos+2]
        return float(q[0]), float(q[1])

    def _ball_speed(self):
        v = self.sim.data.qvel[self.sim.ball_qvel:self.sim.ball_qvel+2]
        return float(np.linalg.norm(v))

    def _observe(self):
        sim, session = self.sim, self.session
        robot_xy, robot_yaw = self._ground_pose(self.role)
        other_xy, other_yaw = self._ground_pose(self.other)
        body = sim.model.body(self.role + '_Nao').id
        roll, pitch = torso_roll_pitch(sim.data.xmat[body])
        self._last_observation = self.builder.build(WorldState(
            time_s=session.penalty_time, remaining_time=geometry.PENALTY_SECONDS - session.penalty_time,
            ball_xy=self._ball_xy(), robot_xy=robot_xy, robot_yaw=robot_yaw,
            opponent_xy=other_xy, opponent_yaw=other_yaw, torso_roll=roll, torso_pitch=pitch,
            gyro=tuple(float(v) for v in sim.data.sensor(self.role + '_imu_gyro').data)))
        return self._last_observation

    def _aim_offset(self, target_y):
        """Bearing to the aim point minus bearing to the goal centre."""
        xy, yaw = self._ground_pose('striker')
        target = (geometry.GOAL_LINE_X, -target_y)  # striker's left is world -y
        return geometry.wrap(geometry.bearing(target, xy, yaw) -
                             geometry.bearing(geometry.GOAL_CENTER, xy, yaw))

    # -- stepping ----------------------------------------------------------
    def _tick(self):
        session = self.session
        session.tick()
        if session.state == 'ERROR':
            raise RuntimeError(session.error)
        speed = self._ball_speed()
        if self.kick_time is not None or self.contact_time is not None:
            self.max_ball_speed = max(self.max_ball_speed, speed)
        if self.contact_time is None and self.sim.kick_contact_count:
            self.contact_time = session.penalty_time
        opponent = getattr(self, '_opponent_striker', None)
        if self.role == 'keeper' and opponent is not None and not opponent.kick_released:
            opponent.aim_offset = self._aim_offset(self._opponent_target)
        self.stopped_for = self.stopped_for + self.sim.dt if speed < STOPPED_SPEED else 0.

    def _advance(self, until=None):
        """Tick to the next FSM decision or until `until()` is true."""
        session = self.session
        guard = int(40/self.sim.dt)
        for _ in range(guard):
            if session.state not in ('SETTLING', 'RUNNING'):
                return
            if until is None and session.control_due:
                return
            if until is not None and until():
                return
            self._tick()
        raise RuntimeError('presupuesto de ticks agotado')

    def _step_window(self):
        """Run the decision tick, then advance to the next decision."""
        if self.session.control_due:
            self._tick()
        self._advance()

    def step(self, action):
        try:
            if self.role == 'striker':
                return self._striker_step(action)
            return self._keeper_step(action)
        except (RuntimeError, ValueError, OSError) as error:
            # A native/physics failure ends the episode and is penalised as a
            # fall, so a policy cannot prefer it to a bad outcome. The broken
            # session is discarded; reset() builds a new one.
            self.outcome = 'sim_error'
            self.close()
            return self._last_observation, StepEvents('sim_error', fallen=True),                 {'sim_error': str(error)}

    def _striker_step(self, action):
        striker, session = self.striker, self.session
        details = {}
        if not striker.kick_released:
            striker.aim_offset = self._aim_offset(action.target_y)
            striker.fire = action.fires
            kick = ParameterizedKick(action.kick_side)
            params = KickParameters(**dict(vars(self.kick_parameters), power=action.kick_power))
            self.kick_request = kick.apply(params)
            session.manual.update(self.kick_request)
            self._step_window()
            details['safety'] = 'released' if striker.kick_released else striker.last_rejection
            if striker.kick_released:
                self.kick_time = session.penalty_time
                self.kick_action = action
                self.kick_record_params = kick.record()
                self._advance(until=lambda: self._striker_outcome() is not None or
                              session.penalty_time - self.kick_time > FAST_FORWARD_S)
        else:
            self._step_window()
        outcome = self._striker_outcome()
        if outcome is None and striker.kick_released:
            outcome = 'short'
        if outcome is None and (session.state == 'FINISHED' or striker.state == 'STOP'):
            outcome = 'timeout'
        self.outcome = outcome
        if outcome is not None and self.kick_time is not None:
            details['kick_record'] = self._kick_record()
        return self._observe(), StepEvents(outcome, fallen=self.sim.fallen['striker']), details

    def _striker_outcome(self):
        sim = self.sim
        result = sim.result()
        if result in ('goal', 'wide', 'out', 'blocked'):
            return result
        if sim.fallen['striker']:
            return 'fall'
        if (self.kick_time is not None and self.session._kick_done and self.contact_time is not None
                and self.stopped_for >= STOPPED_FOR_S):
            return 'blocked' if sim.block_contacts else 'short'
        return None

    def _crossing_y(self):
        """Goal-line crossing (or final position) in the striker's convention."""
        crossing = self.sim.goal_crossing
        y = crossing[1] if crossing is not None else self._ball_xy()[1]
        return -y

    def _kick_record(self):
        xy, yaw = self._ground_pose('striker')
        crossing_y = -self.sim.goal_crossing[1] if self.sim.goal_crossing is not None else None
        target = self.kick_action.target_y
        return {'parameters': self.kick_record_params, 'target_y': target,
                'initial_pose': [xy[0], xy[1], yaw], 'contact': self.contact_time is not None,
                'contact_time': self.contact_time, 'ball_speed': self.max_ball_speed,
                'crossing_y': crossing_y, 'target_error': abs(self._crossing_y() - target),
                'result': self._striker_outcome() or 'short', 'fallen': self.sim.fallen['striker'],
                'kick_time': self.kick_time,
                # Release of the kick request to first foot-ball contact.
                'kick_duration': (self.contact_time - self.kick_time) if self.contact_time else None}

    def _keeper_step(self, action):
        keeper, session = self.keeper, self.session
        details = {'unapplied': ['amplitude']}
        early = False
        if action.fires and keeper.block is None:
            self.commit_time = session.penalty_time
            self.commit_side = action.block_side
            early = self.contact_time is None or self._ball_speed() < .15
            self.early_dive = early
            keeper.commit(BLOCK_SIDES[action.block_side])
        self._step_window()
        if keeper.block is not None:
            self._advance(until=lambda: self._keeper_outcome() is not None or
                          session.penalty_time - self.commit_time > FAST_FORWARD_S)
        outcome = self._keeper_outcome()
        if outcome is None and keeper.block is not None:
            outcome = ('timeout' if self.contact_time is None else
                       'save' if self.sim.block_contacts else 'out')
        striker = session.controllers['striker']
        if outcome is None and (session.state == 'FINISHED' or
                                (striker.state == 'STOP' and self.contact_time is None)):
            outcome = 'timeout'
        self.outcome = outcome
        return (self._observe(), StepEvents(outcome, fallen=self.sim.fallen['keeper'] and outcome is not None,
                contact=outcome is not None and self.sim.block_contacts > 0, early_dive=early), details)

    def _keeper_outcome(self):
        sim = self.sim
        contact = sim.block_contacts > 0
        if sim.goal_event == 'goal':
            return 'goal'
        result = sim.result()
        if result == 'blocked':
            return 'save'
        if result in ('wide', 'out'):
            return 'save' if contact else 'out'
        if sim.fallen['keeper'] and self.contact_time is None:
            return 'keeper_fall'
        if self.contact_time is not None and self.stopped_for >= STOPPED_FOR_S and \
                self.session.penalty_time - self.contact_time > 1.:
            return 'save' if contact else 'out'
        return None

    def episode_record(self):
        if self.session is None:
            return {'sim_error': True}
        if self.role == 'striker':
            if self.kick_time is None:
                return {'ball_speed': None, 'target_error': None, 'kick_time': None}
            record = self._kick_record()
            return {k: record[k] for k in ('ball_speed', 'target_error', 'kick_time', 'crossing_y')}
        crossing = self.sim.goal_crossing
        shot_y = crossing[1] if crossing is not None else self._ball_xy()[1]
        reaction = None
        if self.commit_time is not None and self.contact_time is not None and \
                self.commit_time >= self.contact_time:
            reaction = self.commit_time - self.contact_time
        return {'reaction_time': reaction, 'early_dive': self.early_dive,
                'wrong_side': self.commit_side is not None and self.commit_side != side_of(shot_y),
                'contact': self.sim.block_contacts > 0, 'shot_y': shot_y,
                'shot_speed': self.max_ball_speed}


def make_world(role, opponent, **options):
    return BHumanWorld(role, opponent, **options)
