"""Copy completed motion experiment evidence; preserve frozen report/model bytes."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil


root = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'penalty').is_dir() and (parent / 'models').is_dir())
source = root / 'runs/keeper_motion'
target = root / 'docs/validation/vision/keeper_motion_v1'
target.mkdir(parents=True, exist_ok=True)


def copy(src, relative):
    dst = target / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def compress(src, relative):
    dst = target / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as zipped:
            zipped.write(src.read_bytes())
    assert gzip.decompress(dst.read_bytes()) == src.read_bytes()


reports = ('run_manifest', 'safety_report', 'train_report', 'validation_report',
           'fit_report', 'test_report', 'method_audit', 'training_observations',
           'final_audit', 'runtime_final_audit')
for name in reports:
    copy(source / 'v1' / (name + '.json'), name + '.json')
for path in sorted((source / 'v1/safety_traces').rglob('*.gz')):
    copy(path, path.relative_to(source / 'v1'))
copy(source / 'legacy_parity.json', 'legacy_parity.json')
for folder in ('smoke_nominal', 'smoke_legacy', 'watch_corners'):
    for path in sorted((source / folder).iterdir()):
        if path.suffix == '.jsonl':
            compress(path, folder + '/' + path.name + '.gz')
        elif path.suffix == '.json':
            copy(path, folder + '/' + path.name)
for path in sorted((source / 'posture_comparison').rglob('*')):
    if not path.is_file():
        continue
    relative = path.relative_to(source)
    if path.suffix == '.jsonl':
        compress(path, str(relative) + '.gz')
    elif path.suffix in ('.json', '.png', '.py'):
        copy(path, relative)
for name in ('build', 'full_tests', 'unit_tests', 'native_tests', 'training',
             'watch', 'smoke_nominal', 'smoke_legacy', 'posture_nominal', 'posture_knee'):
    path = root / 'runs' / ('keeper_motion_' + name + '.log')
    copy(path, 'logs/' + name + '.log')
copy(Path(__file__), 'archive_evidence.py')
print('Evidence copied:', sum(p.is_file() for p in target.rglob('*')), 'files')
print('Model SHA256:', hashlib.sha256((root / 'models/vision/keeper_motion_v1.json').read_bytes()).hexdigest())
