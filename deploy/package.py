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
    'penalty/robot_detector.py',
    'penalty/opponent_perception.py',
    'penalty/nao_opponent_vision.py',
    'penalty/nao.py',
    'deploy/run.sh',
)


def package(profile_path, output_path, vision_runtime=None):
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
    extra_files = []
    opponent = profile.get('vision', {}).get('opponent', {})
    if opponent.get('detector') == 'bhuman':
        from penalty.nao_opponent_vision import validate_profile as validate_opponent_profile
        validate_opponent_profile(profile)
        if vision_runtime is None:
            raise ValueError('build a Linux x86-64 sensor-only vision host and pass --vision-runtime')
        vision_runtime = pathlib.Path(vision_runtime).resolve()
        header = vision_runtime.read_bytes()[:20]
        if len(header) < 20 or header[:6] != b'\x7fELF\x02\x01' or header[18:20] != b'\x3e\x00':
            raise ValueError('NAO V6 requires a Linux x86-64 ELF vision executable; a Windows .exe is invalid')
        library = vision_runtime.parent / 'libonnxruntime.so.1.10.0'
        original = ROOT / 'heritage/bhuman_robot_perception'
        originals = json.loads((original / 'manifest.json').read_text())
        lib_info = originals['binary_dependencies']['Util/onnxruntime/lib/Linux/libonnxruntime.so.1.10.0']
        if not library.is_file() or hashlib.sha256(library.read_bytes()).hexdigest() != lib_info['sha256']:
            raise ValueError('matching, pinned Linux ONNX Runtime library required beside the host')
        extra_files = [('native/bhuman_vision/bhuman_vision', vision_runtime),
                       ('native/bhuman_vision/libonnxruntime.so.1.10.0', library)]
        for name in ('manifest.json', 'License.md', 'Util/onnxruntime/LICENSE',
                     'Util/onnxruntime/ThirdPartyNotices.txt',
                     'Config/NeuralNets/RobotDetector/4_anchor_boxes_model_no_activation_20230629-220730.onnx'):
            data = (original / name).read_bytes()
            if name in originals['files'] and hashlib.sha256(data).hexdigest() != originals['files'][name]:
                raise ValueError('original vision asset checksum differs: ' + name)
            extra_files.append(('heritage/bhuman_robot_perception/' + name, original / name))
    manifest = {'profile': profile['robot_id'], 'role': profile['role'], 'files': {}}
    profile_data = profile_path.read_bytes()
    manifest['files']['profile.json'] = hashlib.sha256(profile_data).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as output_stream, gzip.GzipFile(filename='', mode='wb', fileobj=output_stream, mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w') as archive:
        for relative, source in [(name, ROOT / name) for name in FILES] + extra_files:
            data = source.read_bytes()
            manifest['files'][relative] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o755 if relative.endswith('.sh') or relative == 'native/bhuman_vision/bhuman_vision' else 0o644
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
    parser.add_argument('--vision-runtime', help='Linux x86-64 vision executable, with its pinned .so beside it')
    args = parser.parse_args()
    print(package(args.profile, args.output, args.vision_runtime))
