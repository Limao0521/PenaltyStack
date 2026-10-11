"""Fit kick_heading_v1 from observer traces and replay keeper decisions offline.

Input: `--frames` JSONL written by evaluate_keeper_fusion.py with the
`observer` keeper (it never commits, so each seed's perception stream is
complete). Evaluation truth in the traces is used only as fitting labels:
the crossing of the free-rolling ball. The fitted map uses the measured
pre-shot heading (visual yaw) exactly as the keeper computes it at runtime.

  python tools/fit_kick_heading.py --frames runs/dev/observer.frames.jsonl \
      --output models/vision/kick_heading_v1.json --report runs/dev/heading_fit.json
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

GOAL_X = -.12


def state_from_frame(f):
    """vision_state_v2 subset rebuilt from a compact trace frame."""
    b, o = f['b'], f['o']
    return {'version': 'vision_state_v2', 'time_s': f['t'], 'capture_s': f['cap'], 'decision_s': f['t'],
            'remaining_time': max(0., 28.-f['t']), 'sensors': {},
            'ball': {'position': (b[0], b[1]), 'velocity': (b[2], b[3]),
                     'velocity_covariance': ((b[4], b[6]), (b[6], b[5])),
                     'position_covariance': ((b[7], 0.), (0., b[8])), 'age_s': b[9],
                     'visible': bool(b[10]), 'position_valid': bool(b[11]), 'velocity_valid': bool(b[12])},
            'opponent': {'position': (o[0], o[1]), 'age_s': o[2], 'yaw': o[3], 'yaw_variance': o[4],
                         'yaw_age_s': o[5], 'yaw_capture_s': o[6], 'position_valid': bool(o[10]),
                         'yaw_valid': bool(o[11]), 'track_id': o[12],
                         'position_covariance': ((.01, 0.), (0., .01))}}


def true_crossing(episode):
    """Free-rolling crossing from evaluation-only ball positions after the kick."""
    frames = episode['frames']
    record = episode['record']
    contact = record.get('contact_time')
    if contact is None:
        return None
    pts = [(f['t'], f['truth'][0], f['truth'][1]) for f in frames
           if f['t'] > contact+.15 and .2 < f['truth'][0] < 1.6]
    if len(pts) < 4:
        return None
    t, x, y = map(np.asarray, zip(*pts))
    if np.ptp(x) < .15:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    speed = np.polyfit(t, x, 1)[0]
    if speed > -.05:
        return None
    return {'y': float(intercept+slope*GOAL_X), 'speed': float(-speed), 'contact': contact}


def v2_projections(frames):
    """(incoming captures, median crossing, median arrival) per frame, as v2 computes them."""
    from penalty.keeper_fusion import FusionKeeper, PoseModel
    from penalty.keeper_vision import median
    gate = FusionKeeper('v5', False, pose_model=PoseModel())
    trajectories, count, last, out = [], 0, None, []
    for f in frames:
        state = state_from_frame(f)
        if last is not None and state['capture_s'] <= last:
            continue
        last = state['capture_s']
        if gate._incoming(state['ball']) is not None:
            trajectories, count = [], 0
            continue
        (bx, by), (vx, vy) = state['ball']['position'], state['ball']['velocity']
        arrival = (GOAL_X-bx)/vx
        trajectories = (trajectories+[(by+vy*arrival, arrival)])[-3:]
        count += 1
        out.append((count, median([p[0] for p in trajectories]), median([p[1] for p in trajectories])))
    return out


def replay(policy, frames):
    """Decisions of a keeper on an observer stream; stops at its commit."""
    from penalty.rl.contracts import KeeperAction
    history = []
    for f in frames:
        action = policy(state_from_frame(f))
        history.append((f['t'], dict(getattr(policy, 'decision', {}))))
        if KeeperAction(**action).fires:
            return {'time': f['t'], 'side': action['block_side'], 'decision': dict(policy.decision), 'history': history}
    return None


def v2_replay(frames):
    """v2 on the same stream (rebuilt vision_state_v1 view of v2 fields)."""
    from penalty.keeper_vision import VisionKeeper
    keeper = VisionKeeper()
    for f in frames:
        state = state_from_frame(f)
        state['version'] = 'vision_state_v1'
        action = keeper(state)
        if action['trigger'] >= .5:
            return {'time': f['t'], 'side': action['block_side'], 'decision': dict(keeper.decision)}
    return None


def main():
    from penalty.keeper_fusion import FusionKeeper, FusionSettings, PoseModel, pointing_y
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='write kick_heading_v1 model here')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    episodes = [json.loads(line) for line in args.frames.read_text(encoding='utf-8').splitlines() if line.strip()]
    settings = FusionSettings()
    rows, v2_rows = [], defaultdict(list)
    for e in episodes:
        truth = true_crossing(e)
        if truth is None:
            continue
        # Pose prior exactly as FusionKeeper freezes it at the v2 onset.
        probe = FusionKeeper('v2', True, settings, PoseModel(sigma=1., ball_sigma=(1.,)))
        onset_prior = None
        for f in e['frames']:
            probe(state_from_frame(f))
            if probe.onset is not None and probe.prior is not None:
                onset_prior = dict(probe.prior, onset=probe.onset)
                break
        # v2 median projection error by incoming captures, ignoring its commit.
        for n, y_ball, arrival in v2_projections(e['frames']):
            v2_rows[min(n, 8)].append(y_ball-truth['y'])
        rows.append({'style': e['style'], 'seed': e['seed'], 'truth_y': truth['y'], 'speed': truth['speed'],
                     'target_y': e['record'].get('target_y'), 'prior': onset_prior})
    usable = [r for r in rows if r['prior'] is not None]
    x = np.array([r['prior']['pointing_y'] for r in usable])
    y = np.array([r['truth_y'] for r in usable])
    # Robust linear fit (Huber IRLS); leave-one-style-out residuals for sigma.
    def huber_fit(x, y, k=.08, steps=50):
        w = np.ones_like(x)
        for _ in range(steps):
            design = np.vstack([np.ones_like(x), x]).T*np.sqrt(w)[:, None]
            coef = np.linalg.lstsq(design, y*np.sqrt(w), rcond=None)[0]
            r = y-(coef[0]+coef[1]*x)
            w = np.where(np.abs(r) <= k, 1., k/np.maximum(np.abs(r), 1e-9))
        return coef
    coef = huber_fit(x, y)
    styles = sorted({r['style'] for r in usable})
    loso = []
    for style in styles:
        train = [i for i, r in enumerate(usable) if r['style'] != style]
        test = [i for i, r in enumerate(usable) if r['style'] == style]
        c = huber_fit(x[train], y[train])
        loso += [float(y[i]-(c[0]+c[1]*x[i])) for i in test]
    loso = np.array(loso)
    sigma = float(1.4826*np.median(np.abs(loso-np.median(loso))))
    rms = float(np.sqrt(np.mean(loso**2)))
    ball_sigma = []
    for n in sorted(v2_rows):
        errors = np.array(v2_rows[n])
        ball_sigma.append(float(max(.02, 1.4826*np.median(np.abs(errors)))))
    report = {'episodes': len(episodes), 'with_truth': len(rows), 'with_prior': len(usable),
              'prior_coverage_by_style': {s: [sum(r['prior'] is not None for r in rows if r['style'] == s),
                                              sum(r['style'] == s for r in rows)] for s in styles},
              'fit': {'intercept': float(coef[0]), 'slope': float(coef[1]), 'loso_robust_sigma': sigma,
                      'loso_rms': rms, 'loso_p90_abs': float(np.quantile(np.abs(loso), .9))},
              'v2_projection_error_robust_sigma_by_count': {str(n): s for n, s in zip(sorted(v2_rows), ball_sigma)},
              'v2_projection_error_samples_by_count': {str(n): len(v2_rows[n]) for n in sorted(v2_rows)},
              'rows': rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=1))
    if args.output:
        model = {'version': 'kick_heading_v1', 'intercept': float(coef[0]), 'slope': float(coef[1]),
                 'sigma': max(.03, sigma), 'ball_sigma': ball_sigma[:8] or [.25],
                 'provenance': 'observer traces %s; %d kicks with frozen pre-shot heading; leave-one-style-out '
                               'robust sigma; simulated camera pipeline only' % (args.frames.name, len(usable))}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(model, indent=1)+'\n', encoding='utf-8')
        print('Modelo: %s' % args.output)


if __name__ == '__main__':
    main()
