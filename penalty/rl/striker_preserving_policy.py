"""A new Gaussian exploration policy preserving the deployed v2 actor.

Only the Gaussian mean used for sampling is transformed.  For every finite
raw mean m, clipping the transformed mean to [-1, 1] gives exactly clip(m).
The published NumPy exporter can therefore keep its original action head
for deterministic deployment.  Its stochastic sampler is NOT equivalent to
this policy and must not collect training transitions for this experiment.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


POLICY_VERSION = 'striker_preserving_gaussian_v1'
INITIALIZATION_VERSION = 'striker_policy_init_v3'
TAIL_SLOPE = .01
INITIAL_STD = .25
MIN_STD = .12
MAX_STD = .40


def compress_action_tails(means):
    """Compress tails with their true derivative; leave [-1, 1] unchanged.

    This remains a Gaussian with transformed means, not a transformed sample
    distribution.  SB3 records the actual sampled actions and their Gaussian
    log probabilities before the usual environment action clipping.
    """
    if not torch.is_floating_point(means):
        raise TypeError('floating point action means required')
    return torch.where(means > 1., 1. + TAIL_SLOPE*(means - 1.),
                       torch.where(means < -1., -1. + TAIL_SLOPE*(means + 1.), means))


class PreservingActorCriticPolicy(ActorCriticPolicy):
    """Four independent Gaussians with deployment-preserving tail means."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self.action_dist) is not DiagGaussianDistribution:
            raise ValueError('preserving policy requires independent Gaussian actions')
        if (self.action_space.shape != (4,) or
                not np.array_equal(self.action_space.low, np.full(4, -1.)) or
                not np.array_equal(self.action_space.high, np.full(4, 1.))):
            raise ValueError('preserving policy requires four actions bounded by [-1, 1]')

    def _get_action_dist_from_latent(self, latent_pi):
        if type(self.action_dist) is not DiagGaussianDistribution:
            raise ValueError('preserving policy requires independent Gaussian actions')
        means = compress_action_tails(self.action_net(latent_pi))
        return self.action_dist.proba_distribution(means, self.log_std)


def _actor_state(policy):
    """Actor features and final head; exclude critic and exploration sigma."""
    return {name: value for name, value in policy.state_dict().items()
            if name != 'log_std' and not name.startswith(
                ('mlp_extractor.value_net.', 'value_net.', 'vf_features_extractor.'))}


def actor_parameters_sha256(policy):
    digest = hashlib.sha256()
    for name, value in sorted(_actor_state(policy).items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode('utf-8') + b'\0')
        digest.update(str(array.dtype).encode('ascii') + b'\0')
        digest.update(str(array.shape).encode('ascii') + b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def clamp_std(model, minimum=MIN_STD, maximum=MAX_STD):
    """Bound the actual diagonal Gaussian sigma after an optimizer update."""
    if (not math.isfinite(minimum) or not math.isfinite(maximum) or
            minimum <= 0. or maximum < minimum):
        raise ValueError('finite positive sigma bounds required')
    if not isinstance(model.policy, PreservingActorCriticPolicy):
        raise ValueError('expected the preserving exploration policy')
    with torch.no_grad():
        model.policy.log_std.clamp_(math.log(minimum), math.log(maximum))


def initialize_preserving_model(base_model_path, env, **ppo_settings):
    """Create fresh PPO for reward v3; copy the v2 actor without calibration.

    The caller supplies an environment with observation normalization already
    frozen to v2.  Neither this helper nor the policy changes normalization.
    Gamma is one for the finite penalty horizon.  Loading subsequent saved
    checkpoints must use PPO.load directly to preserve their optimizer state.
    """
    if 'policy_kwargs' in ppo_settings:
        raise ValueError('policy architecture is taken exactly from the base model')
    if ppo_settings.get('gamma', 1.) != 1. or ppo_settings.get('use_sde', False):
        raise ValueError('new finite-horizon reward requires gamma=1 and Gaussian exploration')
    if str(ppo_settings.get('device', 'cpu')) != 'cpu':
        raise ValueError('this reproducible initializer uses the CPU runtime')
    path = Path(base_model_path)
    if path.is_dir():
        path = path/'model.zip'
    base_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    base = PPO.load(path, device='cpu')
    if (base.observation_space.shape != (19,) or base.action_space.shape != (4,) or
            type(base.policy.action_dist) is not DiagGaussianDistribution):
        raise ValueError('the published 19-field, four-action striker is required')
    if env.observation_space != base.observation_space or env.action_space != base.action_space:
        raise ValueError('warm-start environment must preserve the published observation/action spaces')
    original_state = {name: value.detach().clone() for name, value in base.policy.state_dict().items()}
    actor_before = actor_parameters_sha256(base.policy)
    # These describe the actual saved architecture, including optimizer type.
    policy_kwargs = deepcopy(base.policy._get_constructor_parameters())
    for name in ('observation_space', 'action_space', 'lr_schedule', 'use_sde'):
        policy_kwargs.pop(name, None)
    settings = dict(ppo_settings)
    settings.update(gamma=1., use_sde=False, device='cpu')
    model = PPO(PreservingActorCriticPolicy, env, policy_kwargs=policy_kwargs, **settings)
    model.policy.load_state_dict(original_state, strict=True)
    with torch.no_grad():
        model.policy.value_net.weight.zero_()
        model.policy.value_net.bias.zero_()
        model.policy.log_std.fill_(math.log(INITIAL_STD))
    # PPO construction owns a completely new Adam, not the old reward's moments.
    if model.policy.optimizer.state:
        raise RuntimeError('fresh reward optimizer unexpectedly has inherited state')
    model.num_timesteps = int(base.num_timesteps)
    model.ep_info_buffer = model.ep_success_buffer = None
    actor_after = actor_parameters_sha256(model.policy)
    if actor_before != actor_after:
        raise RuntimeError('initialization changed the deterministic actor parameters')
    allowed = {'log_std', 'value_net.weight', 'value_net.bias'}
    changed = [name for name, value in model.policy.state_dict().items()
               if not torch.equal(value, original_state[name])]
    if not set(changed).issubset(allowed):
        raise RuntimeError('initialization changed parameters outside critic head and sigma')
    if hashlib.sha256(path.read_bytes()).hexdigest() != base_sha:
        raise RuntimeError('base checkpoint changed during initialization')
    return model, {
        'version': INITIALIZATION_VERSION, 'policy_version': POLICY_VERSION,
        'base_model_sha256': base_sha, 'base_actor_sha256': actor_before,
        'initialized_actor_sha256': actor_after, 'actor_parameters_byte_identical': True,
        'action_head_rescaled': False, 'tail_slope': TAIL_SLOPE,
        'deterministic_clipping_identity': 'clip(compress(m), -1, 1) == clip(m, -1, 1) for finite m',
        'tail_derivative': TAIL_SLOPE, 'inside_derivative': 1.,
        'straight_through_gradient': False, 'initial_std': INITIAL_STD,
        'std_bounds': [MIN_STD, MAX_STD], 'gamma': 1.,
        'value_head_zeroed': True, 'optimizer_reset': 'complete',
        'optimizer_initial_state_entries': len(model.policy.optimizer.state),
        'parameters_changed': changed, 'base_timesteps': int(base.num_timesteps),
        'normalization_changed_by_initializer': False,
        'numpy_export_compatible': 'deterministic clipped actions only; raw Gaussian sampling differs',
    }
