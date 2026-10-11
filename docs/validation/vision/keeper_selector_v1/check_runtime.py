"""At most three independent runtime checks; stop at the first commitment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'runs/keeper_selector'
MODEL = ROOT/'models/vision/keeper_selector_v1.json'
REPLAY = BASE/'watch/replay_watch.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    before = digest(MODEL)
    environment = dict(os.environ, **{name: '1' for name in (
        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    attempts = []
    for seed in (77001000001, 77001000002, 77001000003):
        output = BASE/('watch_'+str(seed))
        output.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        command = [sys.executable, str(ROOT/'tools/watch_keeper_selector.py'),
                   '--style', 'mixed', '--seed', str(seed), '--output', str(output)]
        with (output/'watch.log').open('wb') as log:
            watch = subprocess.run(command, cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        metadata = {'seed': seed, 'style': 'mixed', 'returncode': watch.returncode,
                    'wall_s': time.perf_counter()-start, 'model_sha256_before': before,
                    'model_sha256_after': digest(MODEL), 'blas_threads': 1,
                    'purpose': 'runtime verification only; outside blind test; no fitting'}
        metadata['model_intact'] = metadata['model_sha256_before'] == metadata['model_sha256_after']
        write(output/'watch_run.json', metadata)
        if watch.returncode or not metadata['model_intact']:
            raise RuntimeError('runtime check failed; see '+str(output/'watch.log'))
        episode = json.loads((output/'episode.json').read_text(encoding='utf-8'))
        with (output/'replay.log').open('wb') as log:
            replay = subprocess.run([sys.executable, str(REPLAY), '--trace', str(output/'trace.jsonl'),
                                     '--output', str(output/'replay.json')],
                                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if replay.returncode:
            raise RuntimeError('replay failed; see '+str(output/'replay.log'))
        committed = episode.get('commit_time') is not None and episode.get('commit_defense') is not None
        attempts.append({'seed': seed, 'style': 'mixed', 'result': episode['result'],
                         'commit_defense': episode.get('commit_defense'),
                         'commit_time': episode.get('commit_time'), 'committed': committed,
                         'steps': episode['steps'], 'output': str(output.relative_to(ROOT)),
                         'replay_exact_match': json.loads((output/'replay.json').read_text())['exact_match'],
                         'wall_s': metadata['wall_s']})
        print(json.dumps(attempts[-1], allow_nan=False), flush=True)
        if committed:
            break
    initial = json.loads((BASE/'watch/episode.json').read_text(encoding='utf-8'))
    result = {'version': 'keeper_selector_runtime_checks_v1',
              'initial_watch': {'seed': initial['seed'], 'style': 'mixed', 'result': initial['result'],
                                'commit_defense': initial.get('commit_defense'),
                                'commit_time': initial.get('commit_time'), 'steps': initial['steps']},
              'additional_attempts': attempts, 'additional_attempt_count': len(attempts),
              'stopped_at_first_commit': bool(attempts and attempts[-1]['committed']),
              'maximum_additional_attempts': 3, 'model_sha256_before': before,
              'model_sha256_after': digest(MODEL), 'model_intact': digest(MODEL) == before,
              'excluded_from_blind_test': True, 'used_for_fit_or_model_selection': False,
              'script_sha256': digest(Path(__file__))}
    write(BASE/'watch_attempts.json', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
