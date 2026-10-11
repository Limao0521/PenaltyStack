"""Check archived evidence bytes and the bound frozen model using stdlib only."""
import hashlib
import json
from pathlib import Path


folder = Path(__file__).resolve().parent
root = next(parent for parent in folder.parents
            if (parent / 'penalty').is_dir() and (parent / 'models').is_dir())
manifest = json.loads((folder / 'artifacts.json').read_text(encoding='utf-8'))
for relative, expected in manifest['files'].items():
    path = folder / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
model = manifest['frozen_model']
assert hashlib.sha256((root / model['path']).read_bytes()).hexdigest() == model['sha256']
print('Verified %s artifacts and frozen model.' % len(manifest['files']))
