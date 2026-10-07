"""Load a policy directory written by penalty.rl.training.

The directory holds `model.zip` (stable-baselines3 PPO), `obs_norm.npz`
(observation normalization, plain arrays, no pickle) and `config.json`
(role, history length). The policy keeps its own frame history; the caller
must call reset() at the start of each episode (rollout() does).
"""
from collections import deque
import json
from pathlib import Path

import numpy as np


def is_policy_dir(path):
    path = Path(path)
    return (path/'model.zip').is_file() and (path/'config.json').is_file()


class TrainedPolicy:
    def __init__(self, directory, deterministic=True):
        from stable_baselines3 import PPO
        directory = Path(directory)
        self.config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
        self.name = 'trained:%s' % directory
        self.role = self.config['role']
        self.model = PPO.load(directory/'model.zip', device='cpu')
        stats = np.load(directory/'obs_norm.npz')
        self.mean, self.var = stats['mean'], stats['var']
        self.clip, self.epsilon = float(stats['clip_obs']), float(stats['epsilon'])
        self.frames = deque(maxlen=int(self.config.get('history', 1)))
        self.deterministic = deterministic

    def reset(self):
        self.frames.clear()

    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if not self.frames:
            self.frames.extend([observation]*self.frames.maxlen)
        else:
            self.frames.append(observation)
        stacked = np.concatenate(list(self.frames))
        normalized = np.clip((stacked - self.mean)/np.sqrt(self.var + self.epsilon),
                             -self.clip, self.clip)
        action, _ = self.model.predict(normalized.astype(np.float32),
                                       deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)
