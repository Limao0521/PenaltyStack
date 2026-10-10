"""Trained striker against trained keeper in B-Human (cross-play evaluation).

Both tactical controllers decide every 0.1 s FSM period from their own
robot-relative observation, each with its own perception noise; B-Human
executes both. Unlike the single-role environments the episode does not
fast-forward after the kick: the keeper keeps deciding until it commits.

Results: goal (striker wins); save (the keeper touched the ball and it
stopped or went wide); miss (wide, out or short without keeper contact);
striker_fall / keeper_fall (before the shot); timeout; sim_error.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np

from penalty.rl import geometry
from penalty.rl.actors import NumpyPolicy
from penalty.rl.bhuman_world import FAST_FORWARD_S, STOPPED_FOR_S, BHumanWorld
from penalty.rl.contracts import BLOCK_SIDES, KeeperAction, StrikerAction, clip_observation
from penalty.rl.controllers import TacticalKeeperController, TacticalStrikerController
from penalty.rl.metrics import side_summary, wilson_interval
from penalty.rl.observation import ObservationBuilder, WorldState, torso_roll_pitch
from penalty.rl.wrappers import PerceptionNoise
from penalty.skills.kick import KickParameters, ParameterizedKick

RESULTS = ('goal', 'save', 'miss', 'striker_fall', 'keeper_fall', 'timeout', 'sim_error')
CONDITIONS = {
    # Training conditions of ppo_v2/v4: start variation and perception noise.
    'realista': {'world': {'ball_jitter': .05, 'start_jitter': (.05, .1)}, 'noise': True},
    'sin_ruido': {'world': {'ball_jitter': .05, 'start_jitter': (.05, .1)}, 'noise': False},
}


class DuelWorld(BHumanWorld):
    def __init__(self, keeper_gate=False, **options):
        """keeper_gate: the keeper's commit_after_shot safety filter."""
        super().__init__('striker', 'scripted', **options)
        self.keeper_gate = bool(keeper_gate)
        self.keeper_builder = ObservationBuilder()
        self.result = self.kick_action = None

    def _install_controllers(self, rng, options):
        self.striker = TacticalStrikerController(stability=lambda: self._stability('striker'))
        self.keeper = TacticalKeeperController()
        self.session.controllers['striker'] = self.striker
        self.session.controllers['keeper'] = self.keeper

    def reset(self, rng, options=None):
        striker_obs, info = super().reset(rng, dict(options or {}))
        self.keeper_builder.reset()
        self.result = self.kick_action = None
        return striker_obs, self._observe_keeper(), info

    def _observe_keeper(self):
        sim, session = self.sim, self.session
        robot_xy, robot_yaw = self._ground_pose('keeper')
        other_xy, other_yaw = self._ground_pose('striker')
        roll, pitch = torso_roll_pitch(sim.data.xmat[sim.model.body('keeper_Nao').id])
        return self.keeper_builder.build(WorldState(
            time_s=session.penalty_time, remaining_time=geometry.PENALTY_SECONDS - session.penalty_time,
            ball_xy=self._ball_xy(), robot_xy=robot_xy, robot_yaw=robot_yaw,
            opponent_xy=other_xy, opponent_yaw=other_yaw, torso_roll=roll, torso_pitch=pitch,
            gyro=tuple(float(v) for v in sim.data.sensor('keeper_imu_gyro').data)))

    def _result(self):
        sim = self.sim
        if sim.goal_event == 'goal':
            return 'goal'
        outcome = sim.result()
        if outcome == 'blocked':
            return 'save'
        if outcome in ('wide', 'out'):
            return 'save' if sim.block_contacts else 'miss'
        if sim.fallen['striker'] and self.kick_time is None:
            return 'striker_fall'
        if sim.fallen['keeper'] and self.contact_time is None:
            return 'keeper_fall'
        if (self.contact_time is not None and self.stopped_for >= STOPPED_FOR_S
                and self.session.penalty_time - self.contact_time > 1.):
            return 'save' if sim.block_contacts else 'miss'
        return None

    def duel_step(self, striker_action, keeper_action):
        """Returns (striker_obs, keeper_obs, result or None, details)."""
        session, striker, keeper = self.session, self.striker, self.keeper
        details = {}
        try:
            if not striker.kick_released:
                striker.aim_offset = self._aim_offset(striker_action.target_y)
                striker.fire = striker_action.fires
                kick = ParameterizedKick(striker_action.kick_side)
                params = KickParameters(**dict(vars(self.kick_parameters), power=striker_action.kick_power))
                self.kick_request = kick.apply(params)
                session.manual.update(self.kick_request)
                self.kick_action = striker_action
            if keeper_action.fires and keeper.block is None:
                if self.keeper_gate and self.contact_time is None:
                    details['keeper_safety'] = 'waiting_for_shot'
                else:
                    self.commit_time = session.penalty_time
                    self.commit_side = keeper_action.block_side
                    self.early_dive = self.contact_time is None or self._ball_speed() < .15
                    keeper.commit(BLOCK_SIDES[keeper_action.block_side])
            self._step_window()
            if striker.kick_released and self.kick_time is None:
                self.kick_time = session.penalty_time
            if keeper.block is not None:  # nothing left to decide: run to the outcome
                self._advance(until=lambda: self._result() is not None or
                              session.penalty_time - self.commit_time > FAST_FORWARD_S)
            result = self._result()
            if result is None and keeper.block is not None:
                result = ('timeout' if self.contact_time is None else
                          'save' if self.sim.block_contacts else 'miss')
            if result is None and (session.state == 'FINISHED' or
                                   (striker.state == 'STOP' and self.contact_time is None)):
                result = 'timeout'
        except (RuntimeError, ValueError, OSError) as error:
            self.result = 'sim_error'
            self.close()
            return None, None, 'sim_error', {'sim_error': str(error)}
        self.result = result
        return self._observe(), self._observe_keeper(), result, details

    def record(self):
        if self.session is None:
            return {'result': self.result}
        crossing = self.sim.goal_crossing
        action = self.kick_action
        return {'result': self.result, 'kick_time': self.kick_time, 'contact_time': self.contact_time,
                'commit_time': self.commit_time, 'commit_side': self.commit_side,
                'early_dive': self.early_dive, 'crossing_world_y': None if crossing is None else crossing[1],
                'goal_line_y': None if crossing is None else -crossing[1],
                'target_y': None if action is None else action.target_y,
                'kick_side': None if action is None else action.kick_side,
                'kick_power': None if action is None else action.kick_power,
                'striker_fallen': self.sim.fallen['striker'], 'keeper_fallen': self.sim.fallen['keeper']}


def export_trained(directory):
    """Picklable numpy copy of a trained policy directory (torch stays in the parent)."""
    from penalty.rl.async_ppo import export_policy
    from penalty.rl.policies.trained import TrainedPolicy
    policy = TrainedPolicy(directory)
    return {'kind': 'trained', 'path': str(directory), 'role': policy.role,
            'params': export_policy(policy.model.policy), 'mean': policy.mean, 'var': policy.var,
            'clip': policy.clip, 'epsilon': policy.epsilon, 'history': policy.frames.maxlen}


class NumpyTrainedPolicy:
    """TrainedPolicy (deterministic) without torch: history, normalization, mean action."""

    def __init__(self, spec):
        self.spec = spec
        self.policy = NumpyPolicy(spec['params'])
        self.frames = deque(maxlen=spec['history'])

    def reset(self):
        self.frames.clear()

    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if not self.frames:
            self.frames.extend([observation]*self.frames.maxlen)
        else:
            self.frames.append(observation)
        spec = self.spec
        stacked = np.concatenate(list(self.frames))
        normalized = np.clip((stacked - spec['mean'])/np.sqrt(spec['var'] + spec['epsilon']),
                             -spec['clip'], spec['clip']).astype(np.float32)
        return np.clip(self.policy.mean(normalized), -1., 1.).astype(np.float32)


def player_spec(text):
    """'[NAME=]DIRECTORY[:gate]' for a trained policy or a baseline name
    (scripted, geometric, center). ':gate' turns on the keeper's
    commit_after_shot filter. Returns (name, spec)."""
    from penalty.rl.policies.trained import is_policy_dir
    name, _, rest = text.rpartition('=') if '=' in text else ('', '', text)
    path = rest[:-len(':gate')] if rest.endswith(':gate') else rest
    if not is_policy_dir(path):
        return name or path, {'kind': 'baseline', 'name': path}
    from pathlib import Path
    return name or Path(path).name, dict(export_trained(path), gate=rest.endswith(':gate'))


def make_player(spec, role):
    if spec['kind'] == 'trained':
        if spec['role'] != role:
            raise ValueError('la política %s es de %s, no de %s' % (spec['path'], spec['role'], role))
        return NumpyTrainedPolicy(spec)
    from penalty.rl.policies.baselines import make_policy
    return make_policy(spec['name'], role)


def duel_episode(world, striker, keeper, seed, noise=True, attempts=3):
    rng = np.random.default_rng(seed)
    noises = {role: PerceptionNoise() if noise else None for role in ('striker', 'keeper')}
    noise_rng = {role: np.random.default_rng([seed, i]) for i, role in enumerate(('striker', 'keeper'))}

    def view(role, observation):
        array = clip_observation(observation.to_array())
        return noises[role](array, noise_rng[role]) if noises[role] else array

    for attempt in range(attempts):
        try:
            striker_obs, keeper_obs, _ = world.reset(rng, {})
            break
        except (RuntimeError, OSError):
            world.close()
            if attempt == attempts - 1:
                return {'result': 'sim_error', 'seed': seed, 'steps': 0}
    for player in (striker, keeper):
        if hasattr(player, 'reset'):
            player.reset()
    result, steps = None, 0
    while result is None:
        striker_action = StrikerAction.from_normalized(striker(view('striker', striker_obs)))
        keeper_action = KeeperAction.from_normalized(keeper(view('keeper', keeper_obs)))
        striker_obs, keeper_obs, result, _ = world.duel_step(striker_action, keeper_action)
        steps += 1
    return dict(world.record(), seed=seed, steps=steps)


def _job(job):
    world = DuelWorld(keeper_gate=job['keeper'].get('gate', False), seed=job['engine_seed'],
                      **job['world'])
    striker, keeper = make_player(job['striker'], 'striker'), make_player(job['keeper'], 'keeper')
    try:
        return [duel_episode(world, striker, keeper, seed, job['noise']) for seed in job['seeds']]
    finally:
        world.close()


def summarize_duels(records):
    total = len(records)
    counts = {r: sum(rec['result'] == r for rec in records) for r in RESULTS}
    rates = {r + '_rate': counts[r]/total if total else 0. for r in RESULTS}
    return {'episodes': total, 'counts': counts, 'rates': rates,
            'ci95': {key: wilson_interval(counts[key], total) for key in ('goal', 'save')},
            'striker_fall_after_kick_rate': sum(bool(r.get('striker_fallen')) and r['result'] != 'striker_fall'
                                                for r in records)/total if total else 0.,
            'keeper_commit_rate': sum(r.get('commit_time') is not None for r in records)/total if total else 0.,
            'sides': side_summary(records)}


def duel_matrix(strikers, keepers, episodes, seed=0, workers=1, condition='realista', chunk=10):
    """strikers/keepers: {name: spec} with spec {'kind': 'baseline', 'name': ...}
    or export_trained(...) (+ 'gate' for keepers). Every pair plays the same
    episode seeds (paired comparison)."""
    setting = CONDITIONS[condition]
    seeds = [seed*1_000_000 + i for i in range(episodes)]
    jobs = [{'pair': (s, k), 'striker': strikers[s], 'keeper': keepers[k], 'seeds': seeds[i:i+chunk],
             'world': setting['world'], 'noise': setting['noise'], 'engine_seed': seed}
            for s in strikers for k in keepers for i in range(0, episodes, chunk)]
    start = time.perf_counter()
    if workers <= 1:
        results = [_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_job, jobs))
    by_pair = {}
    for job, records in zip(jobs, results):
        by_pair.setdefault(job['pair'], []).extend(records)
    return {'condition': condition, 'episodes_per_pair': episodes, 'seed': seed,
            'wall_s': time.perf_counter() - start,
            'pairs': {'%s|%s' % pair: summarize_duels(records) for pair, records in by_pair.items()},
            'records': {'%s|%s' % pair: records for pair, records in by_pair.items()}}
