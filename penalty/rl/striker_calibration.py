"""Measured, versioned warm start of saturated striker_v2 action outputs.

Calibration observes v2 acting in the same visual-keeper physics on seeds
outside training/validation/test. It changes only the four action-head rows,
preserving pretrained actor features, value network and observation stats.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


CALIBRATION_VERSION = 'striker_action_calibration_v1'
INITIALIZATION_VERSION = 'striker_policy_init_v2'
ACTION_COMPONENTS = ('target_y', 'kick_side', 'kick_power', 'trigger')
CALIBRATION_BASE = 90001


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_head_scales(means):
    array = np.asarray(means, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4 or not len(array) or not np.isfinite(array).all():
        raise ValueError('finite nonempty action means with shape (n, 4) required')
    p95 = np.quantile(np.abs(array), .95, axis=0)
    scales = np.minimum(1., .75 / np.maximum(p95, .75))
    return {'p95_absolute_mean': p95.tolist(), 'action_head_row_scales': scales.tolist(),
            'raw_mean_clip_fraction': np.mean(np.abs(array) > 1., axis=0).tolist(),
            'scaled_mean_clip_fraction': np.mean(np.abs(array*scales) > 1., axis=0).tolist(),
            'calibration_rows': len(array), 'target_p95_absolute_mean': .75}


def _episode(job):
    from penalty.rl.actors import NumpyPolicy, normalize
    from penalty.rl.striker_visual_env import StrikerVisualKeeperEnv
    spec = job['spec']
    policy = NumpyPolicy(spec['params'])
    stats = (spec['mean'], spec['var'], spec['clip'], spec['epsilon'])
    env = StrikerVisualKeeperEnv(split='train', keeper_mode='balanced')
    rows = []
    try:
        obs, reset_info = env.reset(seed=job['seed'])
        for step in range(500):
            normalized = normalize(obs, stats)
            mean = policy.mean(normalized)
            action = np.clip(mean, -1., 1.).astype(np.float32)
            # Phase bookkeeping is privileged diagnostics, never policy input.
            can_affect_shot = not env.world.striker.kick_released
            rows.append({'step': step, 'pre_release': bool(can_affect_shot),
                         'observation': obs.tolist(), 'normalized_observation': normalized.tolist(),
                         'raw_action_mean': mean.tolist(), 'clipped_action': action.tolist()})
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info['episode_record']['result'] == 'sim_error':
                    raise RuntimeError('calibration physics failure: '+str(info['episode_record']))
                return {'seed': job['seed'], 'reset_info': reset_info,
                        'record': info['episode_record'], 'rows': rows}
        raise RuntimeError('calibration exceeded its bounded episode limit')
    finally:
        env.close()


def calibrate_policy(base_dir, output, episodes=8, workers=4, seed_base=CALIBRATION_BASE):
    from penalty.rl.duel import export_trained
    if (type(episodes) is not int or episodes < 1 or type(workers) is not int or workers < 1
            or seed_base != CALIBRATION_BASE):
        raise ValueError('positive counts and the fixed calibration namespace required')
    base_dir, output = Path(base_dir), Path(output)
    hashes = {name: sha256(base_dir/name) for name in ('config.json', 'model.zip', 'obs_norm.npz')}
    spec = export_trained(base_dir)
    if spec['role'] != 'striker' or spec['history'] != 1 or np.asarray(spec['mean']).shape != (19,):
        raise ValueError('calibration requires the published 19-field striker_v2')
    jobs = [{'spec': spec, 'seed': seed_base*1_000_000+i} for i in range(episodes)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_episode, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print('Calibration %d/%d: seed %d, %s' %
                  (len(rows), episodes, row['seed'], row['record']['result']), flush=True)
    rows.sort(key=lambda row: row['seed'])
    means = [frame['raw_action_mean'] for episode in rows for frame in episode['rows']
             if frame['pre_release']]
    statistics = action_head_scales(means)
    if any(sha256(base_dir/name) != digest for name, digest in hashes.items()):
        raise RuntimeError('the base model changed during calibration')
    report = {'version': CALIBRATION_VERSION, 'initialization': INITIALIZATION_VERSION,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'base_directory': str(base_dir), 'base_hashes': hashes,
              'action_components': list(ACTION_COMPONENTS), 'seed_base': seed_base,
              'episode_seeds': [job['seed'] for job in jobs],
              'percentile_uses_only_pre_release_frames': True,
              'privileged_phase_bookkeeping': 'kick_released is used only to filter calibration diagnostics',
              'normalization_frozen': True, 'validation_or_test_used': False,
              'simulation_only': True, **statistics, 'episodes': rows}
    output.mkdir(parents=True, exist_ok=True)
    path = output/'calibration_report.json'
    if path.exists():
        raise ValueError('choose a new calibration directory; reports are immutable')
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Action head scales:', statistics['action_head_row_scales'], flush=True)
    return report


def _unchanged_parameters_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        if name not in ('action_net.weight', 'action_net.bias'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def apply_head_calibration(model, report):
    """Scale action rows once and discard only their incompatible Adam moments."""
    import torch
    if (report.get('version') != CALIBRATION_VERSION or
            report.get('initialization') != INITIALIZATION_VERSION or
            report.get('validation_or_test_used') is not False or
            report.get('percentile_uses_only_pre_release_frames') is not True or
            report.get('action_components') != list(ACTION_COMPONENTS)):
        raise ValueError('invalid measured head calibration')
    scales = np.asarray(report.get('action_head_row_scales'), dtype=np.float64)
    if scales.shape != (4,) or not np.isfinite(scales).all() or (scales <= 0).any() or (scales > 1).any():
        raise ValueError('invalid action head row scales')
    layer = model.policy.action_net
    if tuple(layer.weight.shape[:1]) != (4,) or tuple(layer.bias.shape) != (4,):
        raise ValueError('striker action head must have four outputs')
    before = _unchanged_parameters_digest(model)
    original_weight = layer.weight.detach().clone()
    original_bias = layer.bias.detach().clone()
    with torch.no_grad():
        tensor = torch.as_tensor(scales, dtype=layer.weight.dtype, device=layer.weight.device)
        layer.weight.mul_(tensor[:, None])
        layer.bias.mul_(tensor)
    for parameter in (layer.weight, layer.bias):
        model.policy.optimizer.state.pop(parameter, None)
    after = _unchanged_parameters_digest(model)
    if before != after:
        raise RuntimeError('head calibration changed actor features or value parameters')
    return {'version': INITIALIZATION_VERSION, 'action_head_row_scales': scales.tolist(),
            'unchanged_parameters_sha256': before, 'actor_features_and_value_unchanged': True,
            'action_head_adam_state_reset': True,
            'max_weight_change': float((layer.weight-original_weight).detach().abs().max()),
            'max_bias_change': float((layer.bias-original_bias).detach().abs().max())}
