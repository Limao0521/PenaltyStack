"""Extracción auditable de los pocos recursos útiles de SabanaHerons2026."""
import argparse
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM_PATHS = (
    'License.md',
    'Config/KickEngine/kickForwardFast.kmc',
    'Config/KickEngine/kickForwardFastLong.kmc',
    'Config/Robots/Default/kickInfo.cfg',
    'Src/Modules/Perception/BallPerceptors/TriondaBallPerceptor.cpp',
    'Src/Modules/Perception/BallPerceptors/TriondaBallPerceptor.h',
    'Src/Modules/Perception/BallPerceptors/TriondaBallSpotsProvider.cpp',
    'Src/Modules/Perception/BallPerceptors/TriondaBallSpotsProvider.h',
)


def migrate(source, commit):
    source = pathlib.Path(source).resolve()
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD']).decode().strip()
    if actual != commit:
        raise ValueError('source commit mismatch: ' + actual)
    destination = ROOT / 'heritage' / 'upstream'
    manifest = {'source': 'https://github.com/juanulloa2050/SabanaHerons2026',
                'commit': commit, 'files': {}}
    for relative in UPSTREAM_PATHS:
        content = subprocess.check_output(['git', '-C', str(source), 'show', commit + ':' + relative])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest['files'][relative] = hashlib.sha256(content).hexdigest()
    (ROOT / 'heritage' / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    migrate(args.source, args.commit)
