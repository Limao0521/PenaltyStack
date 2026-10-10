"""Replay local vision/model JSONL without physical motion or a simulator."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from penalty.keeper_vision import POLICY_VERSION, VisionKeeper
from penalty.perception import BHumanVisionAdapter


def replay(rows):
    keeper, adapter = VisionKeeper(), BHumanVisionAdapter()
    previous_time = None
    for row in rows:
        if row.get('event') == 'STOP':
            keeper.reset()
            adapter.reset()
            yield row
            continue
        packet = row.get('vision', row)
        episode_time = packet.get('time_s', row.get('time_s'))
        if previous_time is not None and episode_time < previous_time:
            keeper.reset()
            adapter.reset()
        if packet.get('version') == 'bhuman_local_models_v1':
            packet = adapter.process(packet, row['received_s'], row['decision_s'], row['time_s'])
        previous_time = packet['time_s']
        action = keeper(packet) if row.get('active', True) else {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        yield {'vision': packet, 'policy': POLICY_VERSION, 'action': action,
               'decision': dict(keeper.decision), 'executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding='utf-8') as source, args.output.open('w', encoding='utf-8') as target:
        for row in replay(json.loads(line) for line in source if line.strip()):
            target.write(json.dumps(row, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
