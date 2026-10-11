"""Learn one original keeper defense at v2's unchanged commitment time.

The camera context contains nine ball measurements. Six logistic heads learn
physical save outcomes, rather than motor trajectories or a new shot trigger.
Normalized crossing and arrival add three quadratic terms; normalization is
fitted on training contexts only. Inference requires NumPy, never Torch.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from penalty.keeper_defenses import ACTION_VERSION, DEFENSE_NAMES, KeeperDefenseAction
from penalty.keeper_vision import POLICY_VERSION as CORE_POLICY_VERSION, VisionKeeper


POLICY_VERSION = 'vision_keeper_selector_v1'
MODEL_VERSION = 'keeper_defense_selector_model_v1'
CONTEXT_VERSION = 'keeper_defense_context_v1'
FEATURE_VERSION = 'keeper_defense_features_v1'
OBJECTIVE_VERSION = 'keeper_defense_outcome_v1'
CONTEXT_FIELDS = ('crossing_y', 'time_to_line', 'ball_x', 'ball_y', 'ball_vx',
                  'ball_vy', 'ball_age_s', 'ball_vx_variance', 'ball_vy_variance')
FEATURE_FIELDS = CONTEXT_FIELDS + ('crossing_z_squared', 'arrival_z_squared',
                                 'crossing_z_times_arrival_z')


def _values(context):
    if not isinstance(context, dict) or context.get('version') != CONTEXT_VERSION:
        raise ValueError('keeper_defense_context_v1 required')
    values = np.asarray(context.get('values'), dtype=float)
    if values.shape != (len(CONTEXT_FIELDS),) or not np.isfinite(values).all():
        raise ValueError('context requires nine finite camera ball values')
    if values[1] < 0 or np.any(values[6:] < 0):
        raise ValueError('arrival, ball age and variances must be nonnegative')
    return values


def context_from_state(state, decision):
    """Read only authorized ball fields and the v2 prediction at commitment."""
    if state.get('version') != 'vision_state_v2':
        raise ValueError('vision_state_v2 required')
    ball = state['ball']
    if not (ball['visible'] and ball['position_valid'] and ball['velocity_valid']):
        raise ValueError('a measured visible ball is required')
    covariance = ball['velocity_covariance']
    context = {'version': CONTEXT_VERSION, 'values': [
        decision['crossing_y'], decision['time_to_line'],
        ball['position'][0], ball['position'][1],
        ball['velocity'][0], ball['velocity'][1], ball['age_s'],
        covariance[0][0], covariance[1][1]]}
    context['values'] = _values(context).tolist()
    return context


extract_context = context_from_state


def _fingerprint(model):
    payload = {key: value for key, value in model.items() if key != 'fingerprint'}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def validate_model(model):
    if not isinstance(model, dict):
        raise ValueError('selector model must be an object')
    expected = {'version': MODEL_VERSION, 'policy': POLICY_VERSION,
                'core_policy': CORE_POLICY_VERSION, 'context_version': CONTEXT_VERSION,
                'feature_version': FEATURE_VERSION, 'objective': OBJECTIVE_VERSION,
                'action_version': ACTION_VERSION}
    if any(model.get(key) != value for key, value in expected.items()):
        raise ValueError('incompatible selector model versions')
    if (model.get('defenses') != list(DEFENSE_NAMES) or
            model.get('context_fields') != list(CONTEXT_FIELDS) or
            model.get('feature_fields') != list(FEATURE_FIELDS) or
            model.get('feature_normalization_clip') != 8.):
        raise ValueError('selector field/defense ordering differs from the contract')
    if (model.get('orientation_used') is not False or
            model.get('learned_joint_trajectories') is not False):
        raise ValueError('selector contract uses ball inputs and original motor primitives only')
    for name, shape in (('mean', (9,)), ('scale', (9,)), ('weights', (6, 12)), ('bias', (6,))):
        values = np.asarray(model.get(name), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('invalid selector %s dimensions/values' % name)
    if np.any(np.asarray(model['scale']) <= 0):
        raise ValueError('selector scale must be positive')
    margin = model.get('tie_margin')
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError('tie_margin must be finite in [0,1]')
    if model.get('fingerprint') != _fingerprint(model):
        raise ValueError('selector fingerprint mismatch')
    return model


def save_model(model, path):
    validate_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def load_model(path):
    return validate_model(json.loads(Path(path).read_text(encoding='utf-8')))


def seal_model(model, training_provenance):
    """Attach immutable run provenance and sign the complete portable artifact."""
    validate_model(model)
    if not isinstance(training_provenance, dict):
        raise ValueError('training provenance must be a JSON object')
    sealed = json.loads(json.dumps(model, allow_nan=False))
    sealed['training_provenance'] = json.loads(json.dumps(training_provenance, allow_nan=False))
    sealed['fingerprint'] = _fingerprint(sealed)
    return validate_model(sealed)


def _features(values, mean, scale):
    z = np.clip((np.asarray(values)-mean)/scale, -8., 8.)
    return np.concatenate((z, np.stack((z[..., 0]**2, z[..., 1]**2,
                                       z[..., 0]*z[..., 1]), axis=-1)), axis=-1)


def _sigmoid(logits):
    return 1./(1.+np.exp(-np.clip(logits, -60., 60.)))


def probabilities(model, context):
    """Six fitted scores; they are experimental, uncalibrated probabilities."""
    features = _features(_values(context), np.asarray(model['mean']), np.asarray(model['scale']))
    return _sigmoid(np.asarray(model['weights']) @ features + np.asarray(model['bias']))


def choose_defense(model, context, original_defense):
    if original_defense not in DEFENSE_NAMES:
        raise ValueError('invalid original defense')
    scores = probabilities(model, context)
    baseline = DEFENSE_NAMES.index(original_defense)
    best = int(np.argmax(scores))
    # A small predicted advantage is insufficient to replace the v2 motion.
    selected = baseline if scores[best]-scores[baseline] <= model['tie_margin']+1e-12 else best
    return DEFENSE_NAMES[selected], scores.tolist()


class _SelectorKeeper:
    def __init__(self, **settings):
        self.ball_keeper = VisionKeeper(**settings)
        self.reset()

    def reset(self):
        self.ball_keeper.reset()
        self.committed_defense = None
        self.context = None
        self.scores = None
        self.context_error = None
        self.decision = dict(self.ball_keeper.decision, policy=POLICY_VERSION,
                             core_policy=CORE_POLICY_VERSION, core_side=0)

    def _select(self, original):
        raise NotImplementedError

    def __call__(self, state):
        try:
            core_state = {'version': 'vision_state_v1' if state.get('version') == 'vision_state_v2' else None}
            for field in ('time_s', 'capture_s', 'remaining_time', 'ball'):
                if field in state:
                    core_state[field] = state[field]
            action = self.ball_keeper(core_state)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            action = self.ball_keeper.wait('invalid_observation')
        if self.ball_keeper.committed is None:
            self.committed_defense = None
            self.context = self.scores = self.context_error = None
        original = {0: 'center', 1: 'left', -1: 'right'}[action['block_side']]
        if action['trigger'] >= .5 and self.committed_defense is None:
            try:
                self.context = context_from_state(state, self.ball_keeper.decision)
            except (AttributeError, KeyError, TypeError, ValueError, IndexError) as error:
                self.context_error = str(error)
                self.context = None
            self.committed_defense = self._select(original)
        defense = self.committed_defense or original
        self.decision = dict(self.ball_keeper.decision, policy=POLICY_VERSION,
                             core_policy=CORE_POLICY_VERSION, core_side=action['block_side'],
                             selected_defense=defense, original_defense=original,
                             pose_prior_used=False, context_version=CONTEXT_VERSION,
                             context_error=self.context_error, scores=self.scores)
        return KeeperDefenseAction(defense, action['trigger'])

    def describe(self):
        core = self.ball_keeper
        return {'policy': POLICY_VERSION, 'core_policy': CORE_POLICY_VERSION,
                'settings': {name: getattr(core, name) for name in (
                    'corner_y', 'dive_lead_s', 'min_incoming_speed', 'confirm_frames',
                    'history', 'max_ball_age', 'sigma_margin', 'goal_line_x')},
                'action_version': ACTION_VERSION, 'context_version': CONTEXT_VERSION,
                'defenses': list(DEFENSE_NAMES), 'inputs': 'camera ball only',
                'orientation_used': False, 'experimental': True,
                'timing': 'unchanged v2 commitment; all six motions eligible'}


class ForcedDefenseKeeper(_SelectorKeeper):
    """Collect a physical counterfactual at the exact v2 commitment time."""
    def __init__(self, defense, **settings):
        if defense not in DEFENSE_NAMES:
            raise ValueError('unknown forced defense')
        self.defense = defense
        super().__init__(**settings)

    def _select(self, original):
        return self.defense

    def describe(self):
        return dict(super().describe(), mode='forced', forced_defense=self.defense)


class LearnedDefenseKeeper(_SelectorKeeper):
    def __init__(self, model, **settings):
        # Keep inference fixed if the caller later edits its JSON dictionary.
        self.model = json.loads(json.dumps(validate_model(model), allow_nan=False))
        super().__init__(**settings)

    def _select(self, original):
        if self.context is None:
            return original
        selected, self.scores = choose_defense(self.model, self.context, original)
        return selected

    def describe(self):
        return dict(super().describe(), mode='learned', model=self.model['fingerprint'],
                    feature_version=FEATURE_VERSION, tie_margin=self.model['tie_margin'])


def _targets(row):
    raw = row.get('targets', row.get('outcomes'))
    if isinstance(raw, dict):
        raw = [raw[name] for name in DEFENSE_NAMES]
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        raise ValueError('six physical outcome targets are required')
    if any(value is not None and value not in (0, 1, 0., 1.) for value in raw):
        raise ValueError('outcome targets must be 0/1/None (sim_error)')
    return np.asarray([0. if value is None else value for value in raw]), np.asarray([value is not None for value in raw])


def _rows(rows):
    rows = list(rows)
    keys = []
    for row in rows:
        if not isinstance(row.get('style'), str) or not row['style']:
            raise ValueError('style is required for balanced training/evaluation')
        if type(row.get('seed')) is not int:
            raise ValueError('integer seed is required for leakage checks')
        keys.append((row['style'], row['seed']))
        _targets(row)
        if row.get('eligible', True) and row.get('context') is not None:
            _values(row['context'])
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate style/seed contexts')
    return rows


def _original(row):
    name = row.get('original_defense')
    if name in DEFENSE_NAMES:
        return name
    decision = row.get('core_decision', {})
    side = decision.get('core_side', decision.get('side'))
    if side in (-1, 0, 1):
        return {0: 'center', 1: 'left', -1: 'right'}[side]
    if row.get('context') is not None:
        crossing = _values(row['context'])[0]
        return 'left' if crossing > .3 else 'right' if crossing < -.3 else 'center'
    return 'center'


def _validation_score(model, rows):
    groups = {}
    counts = {name: 0 for name in DEFENSE_NAMES}
    for row in rows:
        original = _original(row)
        selected = (choose_defense(model, row['context'], original)[0]
                    if row.get('eligible', True) and row.get('context') is not None else original)
        target, measured = _targets(row)
        index = DEFENSE_NAMES.index(selected)
        # Invalid outcomes remain failures in evaluation, even though masked in BCE.
        groups.setdefault(row['style'], []).append(float(target[index]) if measured[index] else 0.)
        counts[selected] += 1
    per_style = {name: float(np.mean(values)) for name, values in groups.items()}
    return {'macro_save_rate': float(np.mean(list(per_style.values()))),
            'worst_style_save_rate': min(per_style.values()), 'per_style': per_style,
            'defense_counts': counts}


def _fit_heads(features, targets, mask, row_weights, regularization, epochs, seed):
    rng = np.random.default_rng(seed)
    weights = rng.normal(0., .001, (6, features.shape[1]))
    observed = row_weights[:, None]*mask
    denominators = observed.sum(0)
    if np.any(denominators <= 0):
        raise ValueError('each defense requires measured physical training outcomes')
    rates = np.clip((observed*targets).sum(0)/denominators, .01, .99)
    bias = np.log(rates/(1-rates))
    mw, vw, mb, vb = np.zeros_like(weights), np.zeros_like(weights), np.zeros(6), np.zeros(6)
    for step in range(1, epochs+1):
        logits = features @ weights.T+bias
        errors = (_sigmoid(logits)-targets)*observed/denominators/6.
        gw = errors.T @ features+regularization*weights
        gb = errors.sum(0)
        mw, vw = .9*mw+.1*gw, .999*vw+.001*gw*gw
        mb, vb = .9*mb+.1*gb, .999*vb+.001*gb*gb
        weights -= .03*(mw/(1-.9**step))/(np.sqrt(vw/(1-.999**step))+1e-8)
        bias -= .03*(mb/(1-.9**step))/(np.sqrt(vb/(1-.999**step))+1e-8)
    loss = np.sum((np.logaddexp(0., features @ weights.T+bias)-targets*(features @ weights.T+bias))*observed/denominators)/6.
    return weights, bias, float(loss)


def fit_model(dataset_rows, validation_rows, *, regularizations=(.001, .01, .1),
              margins=(0., .03, .08), epochs=1200, seed=0):
    """Fit six BCE heads; choose L2/margin using only disjoint validation seeds.

    Rows contain style, seed, a versioned context and six targets (0/1/None).
    ``targets`` may map DEFENSE_NAMES to values; ``outcomes`` may be a list.
    No-context rows are reported/excluded from fitting. Style balances the
    objective and selects models, but never enters camera features or inference.
    """
    train, validation = _rows(dataset_rows), _rows(validation_rows)
    if not train or not validation:
        raise ValueError('nonempty training and validation datasets required')
    if {r['seed'] for r in train} & {r['seed'] for r in validation}:
        raise ValueError('training and validation seeds must be disjoint across all styles')
    if type(epochs) is not int or epochs < 1:
        raise ValueError('positive integer epochs required')
    if (not regularizations or not margins or
            any(not math.isfinite(v) or v < 0 for v in regularizations) or
            any(not math.isfinite(v) or not 0 <= v <= 1 for v in margins)):
        raise ValueError('invalid regularization/margin candidates')
    eligible = [row for row in train if row.get('eligible', True) and row.get('context') is not None]
    if not eligible:
        raise ValueError('no camera commitment contexts to fit')
    values = np.stack([_values(row['context']) for row in eligible])
    mean, scale = values.mean(0), np.maximum(values.std(0), 1e-6)
    features = _features(values, mean, scale)
    targets, mask = zip(*[_targets(row) for row in eligible])
    targets, mask = np.stack(targets), np.stack(mask)
    counts = {style: sum(row['style'] == style for row in eligible) for style in {r['style'] for r in eligible}}
    row_weights = np.asarray([1./counts[row['style']] for row in eligible])
    candidates, selected, best = [], None, None
    for regularization in regularizations:
        weights, bias, loss = _fit_heads(features, targets, mask, row_weights, regularization, epochs, seed)
        for margin in margins:
            model = {'version': MODEL_VERSION, 'policy': POLICY_VERSION,
                     'core_policy': CORE_POLICY_VERSION, 'context_version': CONTEXT_VERSION,
                     'feature_version': FEATURE_VERSION, 'objective': OBJECTIVE_VERSION,
                     'action_version': ACTION_VERSION, 'defenses': list(DEFENSE_NAMES),
                     'context_fields': list(CONTEXT_FIELDS), 'feature_fields': list(FEATURE_FIELDS),
                     'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': weights.tolist(),
                     'bias': bias.tolist(), 'tie_margin': float(margin),
                     'feature_normalization_clip': 8., 'regularization': float(regularization),
                     'orientation_used': False, 'learned_joint_trajectories': False,
                     'experimental': True}
            score = _validation_score(model, validation)
            candidates.append(dict(score, regularization=float(regularization),
                                   tie_margin=float(margin), training_bce=loss))
            # Validation macro save rate, then worst style, then more conservative margin/L2.
            rank = (score['macro_save_rate'], score['worst_style_save_rate'], margin, regularization)
            if best is None or rank > best:
                best, selected = rank, model
    selected['fit_report'] = {
        'training_rows': len(train), 'training_contexts': len(eligible),
        'excluded_no_context': len(train)-len(eligible),
        'masked_error_targets': int((~mask).sum()),
        'measured_targets_by_defense': dict(zip(DEFENSE_NAMES, mask.sum(0).astype(int).tolist())),
        'validation_rows': len(validation), 'training_seed_count': len({r['seed'] for r in train}),
        'validation_seed_count': len({r['seed'] for r in validation}), 'epochs': epochs,
        'optimizer': 'NumPy full-batch Adam; BCE + L2 weights; style-balanced row weights',
        'feature_formula': 'z=clip((camera_context-train_mean)/train_std,-8,8); '
                           'concat(z,crossing_z^2,arrival_z^2,crossing_z*arrival_z)',
        'selection': 'validation macro save rate, worst style, larger fallback margin, larger L2',
        'candidates': candidates, 'selected_validation': _validation_score(selected, validation),
        'sim_error_handling': 'mask BCE labels; evaluation keeps errors in denominator',
        'seed': seed}
    selected['fingerprint'] = _fingerprint(selected)
    return validate_model(selected)
