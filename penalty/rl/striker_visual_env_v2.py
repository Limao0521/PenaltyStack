"""Goal-probability objective for preserving fine-tuning of striker_v2.

Observations, tactical actions and physical opponents are inherited unchanged
from the published visual environment. The new finite-horizon reward has no
step cost or discount preference; simulation failures are fatal diagnostics.
"""
from copy import deepcopy
import traceback

from penalty.rl.contracts import BLOCK_SIDES, StrikerAction, clip_observation
from penalty.rl.striker_visual_env import (
    MAX_EPISODE_STEPS, RESULTS, StrikerVisualKeeperEnv,
)


ENVIRONMENT_VERSION = 'striker_visual_keeper_env_v2'
REWARD_VERSION = 'striker_visual_reward_v3'
SPLIT_BASES = {'train': 101001, 'validation': 102001, 'test': 103001}


def striker_visual_reward_v3(result):
    """Undiscounted +1 for goals, -1 for terminal failures, zero otherwise.

    sim_error is zero-valued diagnostic data, never an optimizable transition.
    The trainer must abort and discard the current rollout upon seeing it.
    With gamma=1 and the observed 28-second horizon, E[return]=2*P(goal)-1.
    """
    if result not in (None, *RESULTS):
        raise ValueError('unknown duel result')
    return 0. if result in (None, 'sim_error') else 1. if result == 'goal' else -1.


class StrikerVisualKeeperEnvV2(StrikerVisualKeeperEnv):
    reward_id = REWARD_VERSION

    def __init__(self, *, split='train', seed_base=None, **kwargs):
        if split not in SPLIT_BASES:
            raise ValueError('unknown episode split')
        super().__init__(split=split,
                         seed_base=SPLIT_BASES[split] if seed_base is None else seed_base,
                         **kwargs)

    def describe(self):
        description = super().describe()
        description.update(environment=ENVIRONMENT_VERSION, reward=REWARD_VERSION,
                           objective='undiscounted probability of an actual goal',
                           required_training_gamma=1., sim_errors_optimizable=False)
        return description

    def step(self, action):
        # Same physical sequence as v1; version reward and preserve error detail.
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
                self._requested_defense = (BLOCK_SIDES[keeper_action.block_side]
                                          if self.keeper_mode == 'scripted' else keeper_action.defense)
            observation, self._keeper_observation, result, details = self.world.duel_step(decoded, keeper_action)
            if result not in (None, *RESULTS):
                raise RuntimeError('unknown physical duel result: '+str(result))
            if result == 'sim_error':
                details = dict(details)
                # Existing worlds can swallow an exception and return its text.
                # Preserve that text; distinguish a call-site stack from a trace.
                details.setdefault('sim_error_origin', 'world.duel_step returned sim_error')
                details.setdefault('sim_error_traceback_available', bool(details.get('sim_error_traceback')))
                if not details['sim_error_traceback_available']:
                    details['sim_error_call_site'] = ''.join(traceback.format_stack())
            if observation is not None:
                self._last_observation = self._observation(observation)
            elif result != 'sim_error':
                raise RuntimeError('physical duel returned no striker observation')
            fallen = (result in ('striker_fall', 'sim_error') or
                      (self.world.session is not None and bool(self.world.sim.fallen['striker'])))
        except Exception as error:
            result, fallen = 'sim_error', True
            details = {'sim_error': str(error), 'sim_error_type': type(error).__name__,
                       'sim_error_traceback': traceback.format_exc(),
                       'sim_error_traceback_available': True,
                       'sim_error_origin': 'exception propagated to environment'}
        reward = float(striker_visual_reward_v3(result))
        self._steps += 1
        self._return += reward
        self._fallen |= fallen
        terminated, truncated = result is not None, False
        info = dict(details, outcome=result, result=result, episode_seed=self.episode_seed,
                    split=self.split, worker_index=self.worker_index,
                    optimizable=result != 'sim_error', keeper_decision=dict(self.keeper.decision))
        if terminated:
            self._done = True
            try:
                record = self._record(result, result, details)
                for name in ('sim_error_type', 'sim_error_traceback', 'sim_error_origin',
                             'sim_error_traceback_available', 'sim_error_call_site'):
                    if name in details:
                        record[name] = details[name]
                record['optimizable'] = result != 'sim_error'
            except Exception:
                self.close()
                raise
            if result == 'sim_error':
                self.close()
                record['children_collected'] = all(process.poll() is not None for process in children)
                info['children_collected'] = record['children_collected']
            info['episode_record'] = record
        return self._last_observation.copy(), reward, terminated, truncated, info
