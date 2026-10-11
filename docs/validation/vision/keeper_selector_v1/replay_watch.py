"""Replay the saved camera packets; this never launches physics or native hosts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = next((parent for parent in Path(__file__).resolve().parents
             if (parent/'penalty').is_dir() and (parent/'models').is_dir()), None)
if ROOT is None:
    raise RuntimeError('cannot locate the repository ancestor containing penalty/ and models/')
sys.path.insert(0, str(ROOT))

from penalty.keeper_library import VisionLibraryKeeper
from penalty.keeper_selector import LearnedDefenseKeeper, load_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=Path(__file__).with_name('trace.jsonl'))
    parser.add_argument('--model', type=Path, default=ROOT/'models/vision/keeper_selector_v1.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('replay.json'))
    args = parser.parse_args()
    model_before = digest(args.model)
    packets = [json.loads(line) for line in args.trace.read_text(encoding='utf-8').splitlines()]
    metadata = next(packet for packet in packets if packet['type'] == 'METADATA')
    learned = LearnedDefenseKeeper(load_model(args.model))
    original = VisionLibraryKeeper('original')
    errors = []
    counts = {'frames': 0, 'action_mismatches': 0, 'decision_mismatches': 0,
              'v2_core_decision_mismatches': 0, 'v2_trigger_mismatches': 0,
              'learned_trigger_frames': 0, 'v2_trigger_frames': 0}
    first_learned = first_original = None
    prediction = None
    selected_scores_observed = False
    if metadata['model_sha256'] != model_before:
        errors.append({'kind': 'trace_model_sha256_mismatch'})
    for packet in packets:
        if packet['type'] != 'FRAME':
            continue
        index = counts['frames']
        state = packet['vision']
        action = learned(state)
        baseline_action = original(state)
        counts['frames'] += 1
        checks = (('action', action.to_dict(), packet['action'], 'action_mismatches'),
                  ('decision', learned.decision, packet['decision'], 'decision_mismatches'),
                  ('v2_core_decision', learned.ball_keeper.decision,
                   original.ball_keeper.decision, 'v2_core_decision_mismatches'),
                  ('v2_trigger', action.trigger, baseline_action.trigger, 'v2_trigger_mismatches'))
        for kind, actual, expected, counter in checks:
            if actual != expected:
                counts[counter] += 1
                errors.append({'frame': index, 'kind': kind, 'actual': actual, 'expected': expected})
        if 'crossing_y' in learned.ball_keeper.decision:
            prediction = dict(learned.ball_keeper.decision)
        selected_scores_observed |= learned.scores is not None
        if action.fires:
            counts['learned_trigger_frames'] += 1
            if first_learned is None:
                first_learned = {'frame': index, 'time_s': state['time_s'],
                                 'capture_s': state['capture_s'], 'defense': action.defense,
                                 'core_decision': dict(learned.ball_keeper.decision)}
        if baseline_action.fires:
            counts['v2_trigger_frames'] += 1
            if first_original is None:
                first_original = {'frame': index, 'time_s': state['time_s'],
                                  'capture_s': state['capture_s'], 'defense': baseline_action.defense,
                                  'core_decision': dict(original.ball_keeper.decision)}
    first_trigger_equal = ((first_learned is None and first_original is None) or
                           (first_learned is not None and first_original is not None and
                            all(first_learned[key] == first_original[key]
                                for key in ('frame', 'time_s', 'capture_s', 'core_decision'))))
    if not first_trigger_equal:
        errors.append({'kind': 'first_trigger_differs_from_v2'})
    model_after = digest(args.model)
    if model_after != model_before:
        errors.append({'kind': 'model_changed_during_replay'})
    report = {'version': 'keeper_selector_trace_replay_v1', 'seed': metadata['seed'],
              'style': metadata['style'], 'counts': counts, 'errors': errors,
              'exact_match': not errors, 'first_trigger': first_learned,
              'v2_first_trigger': first_original, 'first_trigger_equal': first_trigger_equal,
              'prediction': prediction, 'last_decision': learned.decision,
              'model_scores_exercised': selected_scores_observed,
              'model_sha256_before': model_before, 'model_sha256_after': model_after,
              'model_intact': model_before == model_after,
              'trace_sha256': digest(args.trace), 'replay_script_sha256': digest(Path(__file__)),
              'limitations': ('This seed ended without commitment. The replay verifies exact waiting '
                              'and v2 gating but does not exercise learned defense selection.'
                              if first_learned is None else
                              'Camera-trace replay; no physical NAO equivalence.'),
              'simulator_or_native_hosts_launched_by_replay': False}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'frames': counts['frames'], 'exact_match': report['exact_match'],
                      'first_trigger': first_learned, 'model_intact': report['model_intact'],
                      'errors': errors}, indent=2, allow_nan=False))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
