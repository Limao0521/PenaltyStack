"""Summarize a keeper_fusion_benchmark report; optional observer truth by seed.

  python tools/analyze_keeper_fusion.py --report runs/keeper_fusion/final_70001.json \
      --observer runs/keeper_fusion/observer_70001.frames.jsonl --output runs/keeper_fusion/analysis.json
Truth (free-rolling crossing from the observer traces) is used only to
break results down by where the ball would cross; it never feeds a keeper.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))

BINS = ((0., .15), (.15, .25), (.25, .3), (.3, .35), (.35, .45), (.45, .6), (.6, 2.))
NAMES = {'scripted': 'FSM base', 'vision': 'v2 (sin orientación)', 'pose': 'v3 (orientación actual)',
         'pose_v4': 'v4 (v2 + orientación)', 'ball_v5': 'v5 balón', 'full_v5': 'v5 balón + orientación'}


def load_truth(path):
    from fit_kick_heading import true_crossing
    truth = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            episode = json.loads(line)
            t = true_crossing(episode)
            if t is not None:
                truth[(episode['style'], episode['seed'])] = t['y']
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--observer', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    keepers = [k for k in NAMES if k in report['summary']]
    styles = report['styles']
    out = {'report': args.report.name, 'seed': report['seed'], 'episodes_per_style': report['episodes_per_style']}
    print('| Portero | ' + ' | '.join(styles) + ' | Media | Peor |')
    print('|' + '---|'*(len(styles)+3))
    table = {}
    for k in keepers:
        s = report['summary'][k]
        rates = {st: s['per_style'][st]['counts']['save'] for st in styles}
        n = {st: s['per_style'][st]['episodes'] for st in styles}
        table[k] = {'saves': rates, 'episodes': n, 'macro': s['macro_save_rate'], 'worst': s['worst_style'],
                    'worst_rate': s['worst_style_save_rate'], 'sim_errors': s['sim_errors'],
                    'early_commits': s['early_commits']}
        print('| %s | ' % NAMES[k] + ' | '.join('%d/%d' % (rates[st], n[st]) for st in styles) +
              ' | %.1f %% | %s %.0f %% |' % (100*s['macro_save_rate'], s['worst_style'], 100*s['worst_style_save_rate']))
    out['table'] = table
    print()
    for name, c in report['comparisons'].items():
        lo, hi = c['macro_gain_ci95']
        wins = sum(v['only_first_saves'] for v in c['per_style'].values())
        losses = sum(v['only_second_saves'] for v in c['per_style'].values())
        print('%s: %+.1f pp (IC95 %+.1f a %+.1f); solo el primero ataja %d, solo el segundo %d' % (
            name, 100*c['macro_gain'], 100*lo, 100*hi, wins, losses))
    out['comparisons'] = report['comparisons']
    diag = {}
    for k in keepers:
        records = [r for rs in report['records'][k].values() for r in rs]
        decisions = [r.get('keeper_decision') or {} for r in records]
        diag[k] = {'episodes': len(records),
                   'pose_available': sum(bool(d.get('pose_used')) for d in decisions),
                   'pose_changed_side': sum(d.get('side') is not None and d.get('ball_side') is not None and
                                            d['side'] != d['ball_side'] for d in decisions),
                   'held_centre': sum((d.get('holds') or 0) > 0 for d in decisions),
                   'v3_prior_used': sum(bool(d.get('pose_prior_used')) for d in decisions)}
    out['decision_diagnostics'] = diag
    print('\nDiagnóstico de decisiones:', json.dumps(diag, indent=1))
    if args.observer:
        truth = load_truth(args.observer)
        by_bin = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        zone = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for k in keepers:
            for style, records in report['records'][k].items():
                for r in records:
                    y = truth.get((style, r['seed']))
                    if y is None:
                        continue
                    b = next('%.2f-%.2f' % bounds for bounds in BINS if bounds[0] <= abs(y) < bounds[1])
                    by_bin[k][b][0] += r['result'] == 'save'
                    by_bin[k][b][1] += 1
                    side = r.get('commit_side')
                    if side is not None:
                        action = 'centro' if side == 0 else ('lado correcto' if side*y > 0 else 'lado contrario')
                        zone[k][(b, action)][0] += r['result'] == 'save'
                        zone[k][(b, action)][1] += 1
        out['truth_seeds'] = len(truth)
        out['saves_by_crossing'] = {k: {b: v for b, v in sorted(bins.items())} for k, bins in by_bin.items()}
        out['saves_by_crossing_and_action'] = {k: {'%s %s' % key: v for key, v in sorted(z.items())} for k, z in zone.items()}
        print('\nAtajadas por |cruce| real (observador):')
        labels = ['%.2f-%.2f' % b for b in BINS]
        print('| Portero | ' + ' | '.join(labels) + ' |')
        print('|' + '---|'*(len(labels)+1))
        for k in keepers:
            print('| %s | ' % NAMES[k] + ' | '.join('%d/%d' % tuple(by_bin[k][b]) if by_bin[k][b][1] else '-' for b in labels) + ' |')
    if args.output:
        args.output.write_text(json.dumps(out, indent=1)+'\n', encoding='utf-8')
        print('\nAnálisis: %s' % args.output)


if __name__ == '__main__':
    main()
