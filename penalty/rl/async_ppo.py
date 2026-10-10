"""Asynchronous rollout collection for PPO over slow, uneven B-Human episodes.

SubprocVecEnv steps every environment in lockstep, so each vector step waits
for the slowest one; after a kick the ball roll-out (or an episode reset)
lasts as long as dozens of normal steps, and the other environments idle.
Here each worker process (penalty.rl.actors) runs its own environment with a
numpy copy of the current policy and streams transitions. The learner fills
one rollout buffer once enough have arrived and trains with the unchanged
stable-baselines3 PPO update. Collection stays on-policy: workers pause while
the policy updates and resume from where their episodes were.
"""
import math
from multiprocessing import get_context
from multiprocessing.connection import wait
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.torch_layers import FlattenExtractor
import torch as th

from penalty.rl.actors import worker

READY_TIMEOUT_S = 600
WAIT_S = 30


def _linear_tanh(sequential):
    layers = list(sequential)
    if (len(layers) % 2 or not all(isinstance(m, th.nn.Linear) for m in layers[::2])
            or not all(isinstance(m, th.nn.Tanh) for m in layers[1::2])):
        raise ValueError('la recolección asíncrona solo admite MLP Linear+Tanh')
    return [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy()) for m in layers[::2]]


def export_policy(policy):
    """Numpy parameters of an SB3 ActorCriticPolicy for penalty.rl.actors.NumpyPolicy."""
    if (policy.squash_output or policy.use_sde or not isinstance(policy.action_dist, DiagGaussianDistribution)
            or not isinstance(policy.features_extractor, FlattenExtractor)
            or not policy.share_features_extractor):
        raise ValueError('la recolección asíncrona solo admite MlpPolicy gaussiana sin gSDE')
    head = lambda m: [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())]  # noqa: E731
    return {'pi': _linear_tanh(policy.mlp_extractor.policy_net),
            'vf': _linear_tanh(policy.mlp_extractor.value_net),
            'action': head(policy.action_net), 'value': head(policy.value_net),
            'log_std': policy.log_std.detach().cpu().numpy()}


class AsyncCollector:
    def __init__(self, specs):
        context = get_context('spawn')
        self.conns, self.processes = [], []
        try:
            for spec in specs:
                parent, child = context.Pipe()
                process = context.Process(target=worker, args=(child, spec), daemon=True)
                process.start()
                child.close()
                self.conns.append(parent)
                self.processes.append(process)
            for conn in self.conns:
                if not conn.poll(READY_TIMEOUT_S):
                    raise RuntimeError('un worker no inició en %d s' % READY_TIMEOUT_S)
                self._expect(conn, 'ready')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(conn):
        try:
            message, payload = conn.recv()
        except EOFError:
            raise RuntimeError('un worker de recolección terminó inesperadamente') from None
        if message == 'error':
            raise RuntimeError('worker de recolección falló:\n' + payload)
        return message, payload

    @classmethod
    def _expect(cls, conn, kind):
        message, payload = cls._receive(conn)
        if message != kind:
            raise RuntimeError('se esperaba %s de un worker y llegó %s' % (kind, message))
        return payload

    def collect(self, model, normalizer, callback, n_transitions):
        """Fill model.rollout_buffer with >= n_transitions on-policy transitions.
        Returns False when the callback stops training (SB3 semantics)."""
        payload = {'policy': export_policy(model.policy), 'gamma': model.gamma,
                   'obs': (normalizer.obs_rms.mean, normalizer.obs_rms.var,
                           normalizer.clip_obs, normalizer.epsilon)}
        segments = [[] for _ in self.conns]
        callback.on_rollout_start()
        for conn in self.conns:
            conn.send(('run', payload))
        count, keep_going = 0, True
        while count < n_transitions and keep_going:
            for conn in wait(self.conns, timeout=WAIT_S):
                message, data = self._receive(conn)
                if message == 't':
                    segments[self.conns.index(conn)].append(data)
                    count += 1
                    keep_going &= self._on_transition(model, callback, data)
                elif message == 'e':
                    keep_going &= self._on_evaluation(callback, data)
                else:
                    raise RuntimeError('mensaje inesperado de un worker: %s' % message)
        bootstrap = []
        for conn in self.conns:
            conn.send(('pause', None))
        for index, conn in enumerate(self.conns):
            while True:  # transitions still in flight belong to this rollout
                if not conn.poll(READY_TIMEOUT_S):
                    raise RuntimeError('un worker no respondió a la pausa')
                message, data = self._receive(conn)
                if message == 't':
                    segments[index].append(data)
                    keep_going &= self._on_transition(model, callback, data)
                elif message == 'e':
                    keep_going &= self._on_evaluation(callback, data)
                elif message == 'paused':
                    bootstrap.append(data['value'])
                    break
                else:
                    raise RuntimeError('mensaje inesperado de un worker: %s' % message)
        if not keep_going:
            return False
        rows = []
        for segment, value in zip(segments, bootstrap):
            if not segment:
                continue
            # Segments of different workers are concatenated in one column:
            # cut GAE between them and bootstrap a segment that stops mid-episode.
            segment[0]['episode_start'] = True
            if not segment[-1]['done']:
                segment[-1]['reward'] += model.gamma*value
            rows += segment
        model.rollout_buffer = buffer = RolloutBuffer(
            len(rows), model.observation_space, model.action_space, device=model.device,
            gae_lambda=model.gae_lambda, gamma=model.gamma, n_envs=1)
        buffer.observations[:, 0] = np.stack([r['obs'] for r in rows])
        buffer.actions[:, 0] = np.stack([r['action'] for r in rows])
        buffer.rewards[:, 0] = [r['reward'] for r in rows]
        buffer.episode_starts[:, 0] = [r['episode_start'] for r in rows]
        buffer.values[:, 0] = [r['value'] for r in rows]
        buffer.log_probs[:, 0] = [r['log_prob'] for r in rows]
        buffer.pos, buffer.full = len(rows), True
        buffer.compute_returns_and_advantage(last_values=th.zeros(1), dones=np.ones(1))
        normalizer.obs_rms.update(np.stack([r['raw'] for r in rows]))
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    @staticmethod
    def _on_evaluation(callback, evaluation):
        """A deterministic episode: callbacks see it, PPO and timesteps do not."""
        callback.update_locals({'infos': [evaluation], 'dones': np.array([False])})
        return callback.on_step()

    @staticmethod
    def _on_transition(model, callback, transition):
        model.num_timesteps += 1
        infos, dones = [transition['info']], np.array([transition['done']])
        callback.update_locals({'infos': infos, 'dones': dones})
        keep_going = callback.on_step()
        model._update_info_buffer(infos, dones)
        return keep_going

    def close(self):
        """Workers close their environments (and B-Human engines) themselves;
        keep draining so none stays blocked sending a transition."""
        for conn in self.conns:
            try:
                conn.send(('close', None))
            except (OSError, ValueError):
                pass
        deadline = time.monotonic() + 20
        open_conns = list(self.conns)
        while open_conns and any(p.is_alive() for p in self.processes) and time.monotonic() < deadline:
            for conn in wait(open_conns, timeout=.2):
                try:
                    conn.recv()
                except (EOFError, OSError):
                    open_conns.remove(conn)
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for conn in self.conns:
            conn.close()
        self.conns, self.processes = [], []


class AsyncPPO(PPO):
    """PPO whose rollouts come from an AsyncCollector; n_steps is the number
    of transitions per update across all workers. max_std caps the Gaussian
    exploration noise before every rollout (actions live in [-1, 1])."""

    collector = None
    max_std = None

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['collector', 'max_std']

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if self.max_std is not None:
            with th.no_grad():
                self.policy.log_std.clamp_(max=math.log(self.max_std))
        return self.collector.collect(self, self.get_vec_normalize_env(), callback, self.n_steps)
