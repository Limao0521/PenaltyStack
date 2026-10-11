"""Transfer-learning environment against the frozen visual v2 balanced keeper.

The striker retains the published 19-field ``rl_obs_v1`` scene observation
with PerceptionNoise v1. The keeper receives rendered RGB and its own
simulated encoder/IMU/FSR camera transforms, never those scene observations.
Tactical actions are unchanged; a new goal-only terminal reward prevents
timeouts from scoring better than a saved shot. B-Human executes the
original walking, kicking and guarding motions with free-root physics.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from penalty.keeper_defenses import ACTION_VERSION, DEFENSE_CATALOG_VERSION, DEFENSE_NAMES
from penalty.keeper_library import VisionLibraryKeeper
from penalty.rl.contracts import (OBS_HIGH, OBS_LOW, StrikerAction,
                                  KeeperAction, BLOCK_SIDES, clip_observation, versions)
from penalty.rl.defense_benchmark import SensorDefenseDuelWorld
from penalty.rl.duel import DuelWorld
from penalty.rl.policies.baselines import ScriptedKeeperPolicy
from penalty.rl.wrappers import PerceptionNoise
from penalty.sim_orientation import PIPELINE_VERSION, SensorBHumanKeeperCamera


ENVIRONMENT_VERSION = 'striker_visual_keeper_env_v1'
REWARD_VERSION = 'striker_visual_reward_v2'
SPLIT_BASES = {'train': 91001, 'validation': 92001, 'test': 93001}
MAX_EPISODE_STEPS = 500
RESULTS = ('goal', 'save', 'miss', 'striker_fall', 'keeper_fall', 'timeout', 'sim_error')
PERCEPTION = {
    'striker': 'privileged scene rl_obs_v1 plus PerceptionNoise v1',
    'keeper': 'rendered RGB; own simulated encoders/IMU/FSR and estimated contact odometry',
    'limitations': 'ideal own simulated sensors; physics pauses during RGB inference; '
                   'simulation only, no physical NAO equivalence',
}


def striker_visual_reward_v2(result):
    """Goal +10; every other terminal result -10; each decision costs .02.

    The observed 28-second limit belongs to this finite-horizon task. Its
    timeout is a terminal failure rather than a truncation to bootstrap.
    Published striker_reward_v1 remains untouched.
    """
    if result not in (None, *RESULTS):
        raise ValueError('unknown duel result')
    return -.02 + (0. if result is None else 10. if result == 'goal' else -10.)


class _FrozenScriptedKeeper:
    """Our scene-based FSM equivalent; this is not official B-Human behavior."""
    def __init__(self):
        self.policy = ScriptedKeeperPolicy()
        self.reset()

    def reset(self):
        self.decision = {'reason': 'waiting_for_shot', 'mode': 'scripted'}

    def describe(self):
        return {'policy': 'scripted_keeper', 'mode': 'scripted',
                'inputs': 'privileged scene rl_obs_v1 with PerceptionNoise v1',
                'official_bhuman_behavior': False, 'original_bhuman_motion': True}

    def __call__(self, observation):
        action = KeeperAction.from_normalized(self.policy(observation))
        self.decision = {'reason': 'committed' if action.fires else 'waiting_for_shot',
                         'core_side': action.block_side, 'mode': 'scripted',
                         'selected_defense': BLOCK_SIDES[action.block_side]}
        return action


def scheduled_episode_seed(split, episode_index, seed_base=None):
    """Disjoint default split namespaces, with an explicit reproducible override."""
    if split not in SPLIT_BASES:
        raise ValueError('unknown episode split')
    base = SPLIT_BASES[split] if seed_base is None else seed_base
    if (type(base) is not int or not 0 < base < 2**31 or
            type(episode_index) is not int or not 0 <= episode_index < 1_000_000):
        raise ValueError('seed base and episode index must be bounded integers')
    return base * 1_000_000 + episode_index


class StrikerVisualKeeperEnv(gym.Env):
    """A 19-observation/4-action Gymnasium environment compatible with striker_v2.

    ``reset(seed=s)`` uses exactly ``s`` as the physical trial/noise seed.
    Unseeded resets use ``base*1e6 + episode_start + worker_index + n*workers``.
    Explicit seeds do not advance that schedule. World/camera factories are
    dependency-injection hooks for pure API tests, not alternative backends.
    """
    metadata = {'render_modes': []}
    role = 'striker'
    backend = 'bhuman'
    opponent = 'vision_v2_balanced'
    reward_id = REWARD_VERSION

    def __init__(self, *, split='train', seed_base=None, worker_index=0,
                 worker_count=1, episode_start=0, perception_noise=True,
                 keeper_mode='balanced', motion_model=None, world_options=None,
                 world_factory=None, camera_factory=None):
        if keeper_mode not in ('balanced', 'motion', 'scripted'):
            raise ValueError('unknown frozen keeper mode')
        if motion_model is not None and keeper_mode != 'motion':
            raise ValueError('a motion profile requires keeper_mode=motion')
        if (type(worker_count) is not int or worker_count < 1 or
                type(worker_index) is not int or not 0 <= worker_index < worker_count):
            raise ValueError('invalid worker index/count')
        scheduled_episode_seed(split, episode_start + worker_index, seed_base)
        if type(perception_noise) is not bool:
            raise ValueError('perception_noise must be boolean')
        options = {'ball_jitter': .05, 'start_jitter': (.05, .1)}
        options.update(dict(world_options or {}))
        if any(name in options for name in ('simulator_factory', 'keeper_gate', 'commit_after_shot')):
            raise ValueError('the visual keeper backend and timing are frozen')
        self.keeper_mode = keeper_mode
        self.motion_profile = None
        factory = SensorDefenseDuelWorld
        if keeper_mode == 'motion':
            from penalty.motion_contract import ROOT
            from penalty.rl.motion_training import load_profile
            from penalty.rl.motion_world import MotionDuelWorld
            from penalty.skills.keeper_motion import MotionParameters
            path = Path(motion_model or ROOT/'models/vision/keeper_motion_v1.json').resolve()
            profile = load_profile(path)
            parameters = MotionParameters(**profile['parameters'])
            self.motion_profile = {'path': str(path),
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                   'model_version': profile['version'],
                                   'parameters': dict(profile['parameters'])}
            factory = MotionDuelWorld
            options['motion_parameters'] = parameters
        elif keeper_mode == 'scripted':
            factory = DuelWorld
        self.world = (world_factory or factory)(**options)
        self.world_options = options
        self.camera_factory = camera_factory or SensorBHumanKeeperCamera
        self.camera = None
        self.keeper = (_FrozenScriptedKeeper() if keeper_mode == 'scripted'
                       else VisionLibraryKeeper('balanced'))
        self.opponent = 'scripted_fsm' if keeper_mode == 'scripted' else 'vision_v2_'+keeper_mode
        self.control_dt = self.world.control_dt
        self.split, self.seed_base = split, seed_base
        self.worker_index, self.worker_count = worker_index, worker_count
        self.episode_start, self._episode_ordinal = episode_start, 0
        self.perception_noise = perception_noise
        self.noise = PerceptionNoise() if perception_noise else None
        self.keeper_noise = PerceptionNoise() if keeper_mode == 'scripted' else None
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
        self._done = True
        self.episode_seed = None
        self._last_observation = np.zeros(OBS_LOW.shape, dtype=np.float32)
        self._reset_episode_state()

    def _reset_episode_state(self):
        self._steps, self._return, self._fallen = 0, 0., False
        self._visible, self._velocities = 0, 0
        self._first_trigger = self._requested_defense = None
        self._runtime_provenance = None
        self._reset_info = None
        self._keeper_observation = None

    def describe(self):
        perception = dict(PERCEPTION)
        if not self.perception_noise:
            perception['striker'] = 'privileged scene rl_obs_v1; no perception noise'
        if self.keeper_mode == 'scripted':
            perception['keeper'] = 'privileged scene rl_obs_v1 plus independent PerceptionNoise v1; our FSM baseline'
        return dict(versions(), environment=ENVIRONMENT_VERSION, role=self.role,
                    backend=self.backend, opponent=self.opponent, reward=self.reward_id,
                    keeper_mode=self.keeper_mode, finite_horizon_s=28., timeout_is_terminal=True,
                    keeper_action=(versions()['keeper_action'] if self.keeper_mode == 'scripted' else ACTION_VERSION),
                    keeper_policy=self.keeper.describe(),
                    camera_pipeline=None if self.keeper_mode == 'scripted' else PIPELINE_VERSION,
                    perception=perception, hardware_validated=False,
                    learned_joint_trajectories=False, keeper_motion_profile=deepcopy(self.motion_profile))

    def _observation(self, observation):
        array = clip_observation(observation.to_array())
        return self.noise(array, self.noise_rng) if self.noise is not None else array

    def _ensure_camera(self):
        model = self.world.sim.model
        if self.camera is not None and self.camera.model is not model:
            self.camera.close()
            self.camera = None
        if self.camera is None:
            self.camera = self.camera_factory(self.world)
        return self.camera

    def _processes(self):
        processes = []
        session = self.world.session
        if session is not None:
            processes.extend(bridge.process for bridge in session.sim.bridges.values())
        detector = getattr(self.camera, 'detector', None)
        if detector is not None and getattr(detector, 'process', None) is not None:
            processes.append(detector.process)
        return processes

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            index = self.episode_start + self.worker_index + self._episode_ordinal*self.worker_count
            actual_seed = scheduled_episode_seed(self.split, index, self.seed_base)
            self._episode_ordinal += 1
        else:
            if type(seed) is not int or seed < 0:
                raise ValueError('episode seed must be a nonnegative integer')
            actual_seed = seed
        self.episode_seed = actual_seed
        self._done = True
        self._reset_episode_state()
        try:
            observation, self._keeper_observation, info = self.world.reset(
                np.random.default_rng(actual_seed), dict(options or {}))
            if self.keeper_mode != 'scripted':
                hello = self.world.sim.bridges['keeper'].hello
                if (DEFENSE_CATALOG_VERSION not in hello.get('capabilities', ()) or
                        set(hello.get('keeper_defenses', ())) != set(DEFENSE_NAMES)):
                    raise RuntimeError('six-defense native runtime capability required')
                self._ensure_camera().reset()
            self.keeper.reset()
            self.noise_rng = np.random.default_rng([actual_seed, 0])
            if self.noise is not None:
                self.noise.reset()
            if self.keeper_noise is not None:
                self.keeper_noise.reset()
                self.keeper_noise_rng = np.random.default_rng([actual_seed, 1])
            self._last_observation = self._observation(observation)
            self._reset_info = deepcopy(info)
            self._done = False
            result_info = dict(info, **self.describe(), episode_seed=actual_seed,
                               seed=actual_seed, split=self.split, worker_index=self.worker_index)
            return self._last_observation.copy(), result_info
        except Exception:
            self.close()
            raise

    def _record(self, result, outcome, details):
        record = dict(self.world.record(), result=result, outcome=outcome,
                      seed=self.episode_seed, episode_seed=self.episode_seed,
                      split=self.split, worker_index=self.worker_index,
                      steps=self._steps, fallen=self._fallen,
                      duration_s=float(self._last_observation[0]),
                      visible_frames=self._visible, velocity_frames=self._velocities,
                      keeper_decision=dict(self.keeper.decision),
                      first_trigger=deepcopy(self._first_trigger),
                      requested_defense=self._requested_defense,
                      no_commit=self._first_trigger is None,
                      reset_info=deepcopy(self._reset_info),
                      runtime_provenance=deepcopy(self._runtime_provenance),
                      environment=self.describe())
        record['return'] = self._return
        if details.get('sim_error'):
            record['sim_error'] = details['sim_error']
        return record

    def step(self, action):
        if self._done:
            raise RuntimeError('call reset() before step() after an episode ends')
        decoded = StrikerAction.from_normalized(action)
        children = self._processes()
        details = {}
        observation = None
        try:
            if self._steps >= MAX_EPISODE_STEPS:
                raise RuntimeError('visual duel exceeded the bounded decision limit')
            if self.keeper_mode == 'scripted':
                view = self.keeper_noise(clip_observation(self._keeper_observation.to_array()),
                                         self.keeper_noise_rng)
                keeper_action = self.keeper(view)
                now = float(self._last_observation[0])
                state = {'time_s': now, 'capture_s': now,
                         'sensor_provenance': 'privileged scene observation with independent PerceptionNoise v1',
                         'inference_clock': 'scene observation at each simulated tactical step'}
            else:
                state = self.camera.observe()
                keeper_action = self.keeper(state)
                self._visible += bool(state['ball']['visible'])
                self._velocities += bool(state['ball']['velocity_valid'])
            if self._runtime_provenance is None:
                self._runtime_provenance = {
                    'motion': deepcopy(self.world.sim.bridges['keeper'].hello),
                    'robot_detector': deepcopy(getattr(getattr(self.camera, 'detector', None), 'hello', None)),
                    'sensor': deepcopy(state.get('sensor_provenance')),
                    'inference_clock': state.get('inference_clock'),
                }
            if keeper_action.fires and self._first_trigger is None:
                decision = self.keeper.decision
                self._first_trigger = {
                    'time_s': float(state['time_s']), 'capture_s': float(state['capture_s']),
                    'core_side': int(decision['core_side']),
                    'crossing_y': (None if self.keeper_mode == 'scripted' else float(decision['crossing_y'])),
                    'time_to_line': (None if self.keeper_mode == 'scripted' else float(decision['time_to_line'])),
                }
                self._requested_defense = (BLOCK_SIDES[keeper_action.block_side] if self.keeper_mode == 'scripted'
                                           else keeper_action.defense)
            observation, self._keeper_observation, result, details = self.world.duel_step(decoded, keeper_action)
            if result not in (None, *RESULTS):
                raise RuntimeError('unknown physical duel result: '+str(result))
            if observation is not None:
                self._last_observation = self._observation(observation)
            elif result != 'sim_error':
                raise RuntimeError('physical duel returned no striker observation')
            fallen = (result in ('striker_fall', 'sim_error') or
                      (self.world.session is not None and bool(self.world.sim.fallen['striker'])))
        except Exception as error:
            result, fallen = 'sim_error', True
            details = {'sim_error': str(error)}
        outcome = result
        reward = float(striker_visual_reward_v2(result))
        self._steps += 1
        self._return += reward
        self._fallen |= fallen
        terminated, truncated = result is not None, False
        info = dict(details, outcome=outcome, result=result, episode_seed=self.episode_seed,
                    split=self.split, worker_index=self.worker_index,
                    keeper_decision=dict(self.keeper.decision))
        if terminated or truncated:
            self._done = True
            try:
                record = self._record(result, outcome, details)
            except Exception:
                # Provenance/native confirmation failures must not leave the
                # renderer, detector or either motion child behind.
                self.close()
                raise
            if result == 'sim_error':
                self.close()
                record['children_collected'] = all(process.poll() is not None for process in children)
                info['children_collected'] = record['children_collected']
            info['episode_record'] = record
        return self._last_observation.copy(), reward, terminated, truncated, info

    def close(self):
        self._done = True
        camera, self.camera = self.camera, None
        try:
            if camera is not None:
                camera.close()
        finally:
            self.world.close()
