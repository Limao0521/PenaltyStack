"""Shared Gymnasium environment over a synthetic or B-Human world."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from penalty.rl.contracts import OBS_HIGH, OBS_LOW, clip_observation, versions


def _make_world(role, backend, opponent, options):
    if backend == 'synthetic':
        from penalty.rl.synthetic import make_world
    elif backend == 'bhuman':
        from penalty.rl.bhuman_world import make_world
    else:
        raise ValueError('backend desconocido: %s' % backend)
    return make_world(role, opponent, **options)


class PenaltyEnv(gym.Env):
    metadata = {'render_modes': []}
    role = None
    action_type = None
    action_size = None
    default_reward = None
    default_opponent = None

    def __init__(self, backend='synthetic', opponent=None, reward_id=None, backend_options=None):
        from penalty.rl.rewards import get_reward
        self.backend = backend
        self.opponent = opponent or self.default_opponent
        self.reward_id = reward_id or self.default_reward
        self.reward_fn = get_reward(self.reward_id)
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., (self.action_size,), dtype=np.float32)
        self.world = _make_world(self.role, backend, self.opponent, dict(backend_options or {}))
        self.control_dt = self.world.control_dt
        self._steps = 0
        self._return = 0.
        self._fallen = False
        self._done = True

    def describe(self):
        return dict(versions(), role=self.role, backend=self.backend, opponent=self.opponent,
                    reward=self.reward_id)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation, info = self.world.reset(self.np_random, dict(options or {}))
        self._steps, self._return, self._fallen, self._done = 0, 0., False, False
        info.update(self.describe())
        return clip_observation(observation.to_array()), info

    def step(self, action):
        if self._done:
            raise RuntimeError('llama reset() antes de step() tras terminar el episodio')
        decoded = self.action_type.from_normalized(action)
        observation, events, details = self.world.step(decoded)
        reward = float(self.reward_fn(events))
        self._steps += 1
        self._return += reward
        self._fallen |= bool(events.fallen)
        truncated = events.outcome == 'timeout'
        terminated = events.outcome is not None and not truncated
        info = dict(details, outcome=events.outcome)
        if terminated or truncated:
            self._done = True
            record = dict(self.world.episode_record(), outcome=events.outcome, fallen=self._fallen,
                          steps=self._steps, duration_s=float(observation.time_s))
            record['return'] = self._return
            info['episode_record'] = record
        return clip_observation(observation.to_array()), reward, terminated, truncated, info

    def close(self):
        self.world.close()
