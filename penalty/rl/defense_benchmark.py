"""Paired camera/sensor duels for the expanded original keeper defenses.

The published v2 prediction and trigger are unchanged. The new action only
selects which original B-Human guard executes a central commitment.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np

from penalty.keeper_defenses import (ACTION_VERSION, DEFENSE_CATALOG_VERSION,
                                    DEFENSE_NAMES, KeeperDefenseAction, catalog)
from penalty.keeper_library import MODES, VisionLibraryKeeper
from penalty.rl.contracts import KeeperAction, StrikerAction, clip_observation
from penalty.rl.duel import make_player, player_spec
from penalty.rl.keeper_benchmark import STYLES, VariedStrikerPolicy, aggregate, compare_styles
from penalty.rl.wrappers import PerceptionNoise
from penalty.sim_orientation import (PIPELINE_VERSION, BodyCameraKeeperController,
                                     SensorBHumanKeeperCamera, SensorCameraDuelWorld)

BENCHMARK_VERSION = 'keeper_defense_benchmark_v1'


def legacy_side(defense):
    # Existing DuelWorld records the predicted zone, not a new motor index.
    # Both wide guards still replace a CENTRAL v2 request in this experiment.
    return 1 if defense == 'left' else -1 if defense == 'right' else 0


class DefenseCameraKeeperController(BodyCameraKeeperController):
    def __init__(self):
        super().__init__()
        self.requested_defense = None

    def commit(self, side):
        if self.block is not None:
            return
        defense = self.requested_defense
        if defense not in DEFENSE_NAMES:
            raise ValueError('missing named keeper defense')
        expected = {0: 'center', 1: 'left', -1: 'right'}[legacy_side(defense)]
        if side != expected:
            raise ValueError('legacy zone does not match the named defense')
        super().commit(defense)


class SensorDefenseDuelWorld(SensorCameraDuelWorld):
    """Explicit v2-action adapter; the old DuelWorld and its contract stay frozen."""
    def _install_controllers(self, rng, options):
        super()._install_controllers(rng, options)
        self.keeper = DefenseCameraKeeperController()
        self.session.controllers['keeper'] = self.keeper

    def duel_step(self, striker_action, keeper_action):
        if not isinstance(keeper_action, KeeperDefenseAction):
            raise TypeError('keeper_action_v2 required')
        if self.keeper.block is None and keeper_action.fires:
            self.keeper.requested_defense = keeper_action.defense
        adapter = KeeperAction(legacy_side(keeper_action.defense), keeper_action.trigger, .5)
        return super().duel_step(striker_action, adapter)

    def record(self):
        return dict(super().record(), keeper_action=ACTION_VERSION,
                    commit_defense=self.keeper.block if self.session is not None else None)


def camera_episode(world, camera, striker, mode, seed, on_frame=None):
    rng = np.random.default_rng(seed)
    striker_obs, _, _ = world.reset(rng, {})
    hello = world.sim.bridges['keeper'].hello
    if DEFENSE_CATALOG_VERSION not in hello.get('capabilities', ()):
        raise RuntimeError('rebuild motion runtime: keeper_defenses_v2 capability required')
    if set(hello.get('keeper_defenses', ())) != set(DEFENSE_NAMES):
        raise RuntimeError('motion runtime defense catalog differs from Python')
    camera.reset()
    if hasattr(striker, 'reset'):
        striker.reset()
    keeper = VisionLibraryKeeper(mode)
    noise, noise_rng = PerceptionNoise(), np.random.default_rng([seed, 0])
    result, steps, visible, velocities = None, 0, 0, 0
    while result is None:
        state = camera.observe()
        action = keeper(state)
        visible += bool(state['ball']['visible'])
        velocities += bool(state['ball']['velocity_valid'])
        if on_frame is not None:
            on_frame(state, action, dict(keeper.decision))
        striker_action = StrikerAction.from_normalized(
            striker(noise(clip_observation(striker_obs.to_array()), noise_rng)))
        striker_obs, _, result, _ = world.duel_step(striker_action, action)
        steps += 1
    return dict(world.record(), seed=seed, mode=mode, steps=steps,
                visible_frames=visible, velocity_frames=velocities,
                keeper_decision=dict(keeper.decision))


def _job(job):
    world = SensorDefenseDuelWorld(ball_jitter=.05, start_jitter=(.05, .1))
    camera, rows = None, []
    try:
        for seed in job['seeds']:
            # Rotate modes per seed to distribute cold-start effects instead
            # of always evaluating the control before every other defense.
            offset = seed % len(job['modes'])
            modes = job['modes'][offset:] + job['modes'][:offset]
            for mode in modes:
                try:
                    world._ensure_session()
                    if camera is not None and camera.model is not world.sim.model:
                        camera.close()
                        camera = None
                    if camera is None:
                        camera = SensorBHumanKeeperCamera(world)
                    striker = (make_player(job['trained'][job['style']], 'striker')
                               if job['style'] in ('v1', 'v2')
                               else VariedStrikerPolicy(job['style'], seed))
                    record = camera_episode(world, camera, striker, mode, seed)
                except (RuntimeError, ValueError, OSError) as error:
                    record = {'result': 'sim_error', 'seed': seed, 'mode': mode,
                              'sim_error': str(error), 'commit_defense': None,
                              'keeper_action': ACTION_VERSION, 'steps': 0}
                    try:
                        if camera is not None:
                            camera.close()
                    finally:
                        camera = None
                        world.close()
                rows.append((mode, record))
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            world.close()
    return job['style'], rows


def benchmark(modes=MODES, episodes=2, seed=72001, workers=4, styles=STYLES, progress=print):
    modes, styles = tuple(modes), tuple(styles)
    if (episodes < 1 or workers < 1 or not modes or len(set(modes)) != len(modes) or
            any(m not in MODES for m in modes) or not styles or len(set(styles)) != len(styles) or
            any(s not in STYLES for s in styles)):
        raise ValueError('positive episodes/workers and distinct known modes/styles required')
    trained = {name: player_spec('models/rl/striker_'+name)[1]
               for name in ('v1', 'v2') if name in styles}
    seeds = [seed*1_000_000+i for i in range(episodes)]
    jobs = [{'style': style, 'seeds': seeds[i:i+2], 'modes': modes, 'trained': trained}
            for style in styles for i in range(0, episodes, 2)]
    groups = {mode: {style: [] for style in styles} for mode in modes}
    start = time.perf_counter()

    def finish(style, rows, count):
        for mode, record in rows:
            groups[mode][style].append(record)
        if progress:
            progress('[%d/%d] %s: %d duelos (%.1f s)' %
                     (count, len(jobs), style, len(rows), time.perf_counter()-start), flush=True)

    if workers == 1:
        for index, job in enumerate(jobs, 1):
            finish(*_job(job), index)
    else:
        with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
            pending = {pool.submit(_job, job): job for job in jobs}
            for index, future in enumerate(as_completed(pending), 1):
                finish(*future.result(), index)
    for group in groups.values():
        for rows in group.values():
            rows.sort(key=lambda row: row['seed'])
    return {
        'benchmark': BENCHMARK_VERSION, 'camera_pipeline': PIPELINE_VERSION,
        'action_version': ACTION_VERSION, 'catalog': catalog(), 'seed': seed,
        'episode_seeds': seeds, 'episodes_per_style': episodes, 'styles': list(styles),
        'modes': {mode: VisionLibraryKeeper(mode).describe() for mode in modes},
        'wall_s': time.perf_counter()-start,
        'perception': 'rendered RGB, B-Human robot detector, own encoder/IMU/FSR sensor pipeline',
        'prediction': 'unchanged vision_keeper_v2 ball trajectory and commitment timing; orientation unused',
        'sensor_limitations': 'ideal simulated sensor readings, estimated contact odometry; '
                             'physics pauses during camera inference; no hardware equivalence',
        'striker_perception': 'privileged scene observation plus PerceptionNoise v1',
        'order': 'modes rotated per seed; all modes share each style/seed/reset conditions',
        'summary': {mode: aggregate(group) for mode, group in groups.items()},
        'defense_counts': {mode: dict(Counter(row.get('commit_defense') or 'none'
                                           for rows in group.values() for row in rows))
                           for mode, group in groups.items()},
        'comparisons': {mode+'_vs_original': compare_styles(groups[mode], groups['original'])
                        for mode in modes if mode != 'original' and 'original' in groups},
        'records': groups,
    }
