"""Importa sin modificar el subsistema motriz B-Human y su cierre de headers.

El resultado es una referencia versionada para comparar futuras políticas RL.
No afirma que el subsistema compile ni funcione fuera del runtime B-Human.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'baseline' / 'bhuman'
PREFIXES = (
    'Src/Modules/MotionControl/',
    'Src/Modules/Sensing/',
    'Src/Modules/Infrastructure/NaoProvider/',
    'Src/Tools/Motion/',
    'Src/Representations/MotionControl/',
)
CONFIG_NAMES = (
    'armKeyFrameEngine.cfg', 'footOffset.cfg', 'headLimits.cfg',
    'jointCalibration.cfg', 'jointLimits.cfg', 'jointPlayProvider.cfg',
    'keyframeMotionEngine.cfg', 'keyframeMotionParameters.cfg',
    'kickInfo.cfg', 'massCalibration.cfg', 'robotDimensions.cfg',
    'staticJointPoses.cfg', 'walkKickEngine.cfg', 'walkModifier.cfg',
    'walkingEngine.cfg', 'walkingEngineCommon.cfg', 'stiffnessSettings.cfg',
    'replayWalkRequestProvider.cfg', 'odometryDataPreviewProvider.cfg',
    'groundContactDetector.cfg', 'footSupportProvider.cfg',
    'fsrDataProvider.cfg', 'inertialDataProvider.cfg',
    'imuCalibrationProvider.cfg', 'imuCalibration.cfg',
    'energySavingProvider.cfg', 'damageConfigurationBody.cfg',
    'damageConfigurationHead.cfg',
)
INCLUDE = re.compile(rb'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source)] + list(args))


def collect(source):
    source = pathlib.Path(source).resolve()
    commit = git(source, 'rev-parse', 'HEAD').decode().strip()
    tracked = set(git(source, 'ls-tree', '-r', '-z', '--name-only', 'HEAD').decode().strip('\0').split('\0'))
    roots = {path for path in tracked if path.startswith(PREFIXES)}
    roots.update(path for path in tracked if path.startswith('Config/KickEngine/'))
    roots.update(path for path in tracked
                 if path.startswith(('Config/Robots/Default/', 'Config/Robots/Nao/Nao/'))
                 and pathlib.PurePosixPath(path).name in CONFIG_NAMES)
    pending = list(sorted(roots))
    blobs = {}
    while pending:
        relative = pending.pop()
        if relative in blobs:
            continue
        content = git(source, 'show', commit + ':' + relative)
        blobs[relative] = content
        if not relative.startswith('Src/'):
            continue
        for match in INCLUDE.finditer(content):
            name = match.group(1).decode('utf-8', 'replace').replace('\\', '/')
            candidates = ('Src/' + name, str(pathlib.PurePosixPath(relative).parent / name))
            for candidate in candidates:
                if candidate in tracked and candidate not in blobs:
                    pending.append(candidate)
                    break
    return commit, roots, blobs


def migrate(source, expected_commit):
    commit, roots, blobs = collect(source)
    if commit != expected_commit:
        raise ValueError('source commit changed: ' + commit)
    manifest = {'source': 'https://github.com/juanulloa2050/SabanaHerons2026',
                'commit': commit, 'engine_roots': len(roots), 'files': {}}
    for relative, content in sorted(blobs.items()):
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest['files'][relative] = hashlib.sha256(content).hexdigest()
    (ROOT / 'baseline' / 'motion_manifest.json').write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding='utf-8')
    return len(roots), len(blobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        commit, roots, blobs = collect(args.source)
        print('commit=%s roots=%d closure=%d' % (commit, len(roots), len(blobs)))
    else:
        print('roots=%d files=%d' % migrate(args.source, args.commit))
