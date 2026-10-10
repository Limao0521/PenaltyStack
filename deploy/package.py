"""Paquete reproducible y limitado para NAO; no incluye MuJoCo."""
import argparse
import gzip
import hashlib
import io
import json
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = (
    'penalty/__init__.py',
    'penalty/control.py',
    'penalty/vision.py',
    'penalty/perception.py',
    'penalty/keeper_vision.py',
    'penalty/nao_vision.py',
    'penalty/nao.py',
    'deploy/run.sh',
)


def package(profile_path, output_path):
    profile_path = pathlib.Path(profile_path).resolve()
    output_path = pathlib.Path(output_path).resolve()
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    if profile.get('role') not in ('striker', 'keeper'):
        raise ValueError('profile role invalid')
    if not profile.get('robot_id') or 'TO_FILL' in profile['robot_id']:
        raise ValueError('set the physical robot_id before packaging')
    if profile.get('kick_enabled'):
        raise ValueError('physical kick executor is missing')
    if profile.get('keeper_policy') == 'vision':
        from penalty.nao_vision import validate_profile
        validate_profile(profile)
    manifest = {'profile': profile['robot_id'], 'role': profile['role'], 'files': {}}
    profile_data = profile_path.read_bytes()
    manifest['files']['profile.json'] = hashlib.sha256(profile_data).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as output_stream, gzip.GzipFile(filename='', mode='wb', fileobj=output_stream, mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w') as archive:
        for relative in FILES:
            source = ROOT / relative
            data = source.read_bytes()
            manifest['files'][relative] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o755 if relative.endswith('.sh') else 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
        for relative, data in (
            ('profile.json', profile_data),
            ('manifest.json', json.dumps(manifest, sort_keys=True, indent=2).encode('utf-8')),
        ):
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(package(args.profile, args.output))
