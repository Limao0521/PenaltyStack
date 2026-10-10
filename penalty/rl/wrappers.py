"""Observation wrappers: history, delay and perception noise/dropout."""
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from penalty.rl.contracts import FIELD, OBS_HIGH, OBS_LOW


class ObservationHistory(gym.ObservationWrapper):
    """Stacks the last `length` observations, oldest first, flattened."""

    def __init__(self, env, length=6):
        super().__init__(env)
        if length < 1:
            raise ValueError('length debe ser >= 1')
        self.length = length
        low, high = env.observation_space.low, env.observation_space.high
        self.observation_space = spaces.Box(np.tile(low, length), np.tile(high, length),
                                            dtype=np.float32)
        self._frames = deque(maxlen=length)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.length):
            self._frames.append(observation)
        return self.observation(None), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def observation(self, observation):
        return np.concatenate(list(self._frames)).astype(np.float32)


class DelayedObservationWrapper(gym.ObservationWrapper):
    """The policy sees the observation from `delay_steps` decisions ago."""

    def __init__(self, env, delay_steps=1):
        super().__init__(env)
        if delay_steps < 0:
            raise ValueError('delay_steps debe ser >= 0')
        self.delay_steps = delay_steps
        self._queue = deque(maxlen=delay_steps+1)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._queue.clear()
        for _ in range(self.delay_steps+1):
            self._queue.append(observation)
        return self._queue[0], info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._queue.append(observation)
        return self._queue[0], reward, terminated, truncated, info

    def observation(self, observation):
        return observation


class PerceptionNoise:
    """Gaussian ball/opponent/IMU noise and ball dropout for one robot.

    During dropout the ball estimate is held, `ball_age` grows by the
    control period and `ball_visible` is 0.
    """

    def __init__(self, ball_sigma=.03, ball_sigma_per_m=.03, opponent_sigma=.05,
                 gyro_sigma=.02, dropout=.05, control_dt=.1):
        self.ball_sigma, self.ball_sigma_per_m = ball_sigma, ball_sigma_per_m
        self.opponent_sigma, self.gyro_sigma, self.dropout = opponent_sigma, gyro_sigma, dropout
        self.control_dt = control_dt
        self._held = None

    def reset(self):
        self._held = None

    def __call__(self, observation, rng):
        obs = np.array(observation, dtype=np.float32)
        ball = [FIELD[n] for n in ('ball_x', 'ball_y', 'ball_vx', 'ball_vy')]
        if self._held is not None and rng.random() < self.dropout:
            obs[ball] = self._held[ball]
            obs[FIELD['ball_age']] = self._held[FIELD['ball_age']] + self.control_dt
            obs[FIELD['ball_visible']] = 0.
        else:
            distance = float(np.hypot(obs[FIELD['ball_x']], obs[FIELD['ball_y']]))
            sigma = self.ball_sigma + self.ball_sigma_per_m*distance
            obs[ball[:2]] += rng.normal(0, sigma, 2)
        for name in ('opponent_x', 'opponent_y'):
            obs[FIELD[name]] += rng.normal(0, self.opponent_sigma)
        for name in ('gyro_x', 'gyro_y', 'gyro_z'):
            obs[FIELD[name]] += rng.normal(0, self.gyro_sigma)
        obs = np.clip(obs, OBS_LOW, OBS_HIGH)
        self._held = obs.copy()
        return obs


class PerceptionNoiseWrapper(gym.ObservationWrapper):
    """PerceptionNoise on an environment. Uses the env's np_random so the
    noise is reproducible from reset(seed=...)."""

    def __init__(self, env, ball_sigma=.03, ball_sigma_per_m=.03, opponent_sigma=.05,
                 gyro_sigma=.02, dropout=.05):
        super().__init__(env)
        self.noise = PerceptionNoise(ball_sigma, ball_sigma_per_m, opponent_sigma, gyro_sigma, dropout,
                                     getattr(env.unwrapped, 'control_dt', .1))

    @property
    def dropout(self):
        return self.noise.dropout

    def reset(self, **kwargs):
        self.noise.reset()
        return super().reset(**kwargs)

    def observation(self, observation):
        return self.noise(observation, self.env.unwrapped.np_random)
