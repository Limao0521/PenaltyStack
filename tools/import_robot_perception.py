"""Select immutable robot-detector originals from our pinned B-Human revision."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '2360ab1100ad28f70e190c893f6b095651ff0bbc'
UPSTREAM = 'https://github.com/juanulloa2050/SabanaHerons2026'
DEST = ROOT / 'heritage/bhuman_robot_perception'
MODEL = '4_anchor_boxes_model_no_activation_20230629-220730'
FILES = (
    'License.md',
    'Src/Modules/Perception/PlayersPerceptors/RobotDetector.cpp',
    'Src/Modules/Perception/PlayersPerceptors/RobotDetector.h',
    'Src/Libs/ImageProcessing/Resize.cpp', 'Src/Libs/ImageProcessing/Resize.h',
    'Src/Libs/ImageProcessing/LabelImage.cpp', 'Src/Libs/ImageProcessing/LabelImage.h',
    'Src/Libs/ImageProcessing/ColorModelConversions.h',
    'Src/Modules/Perception/ImagePreprocessors/ECImageProvider.cpp',
    'Config/NeuralNets/RobotDetector/4_anchor_boxes_20230629-220730.cfg',
    'Config/NeuralNets/RobotDetector/' + MODEL + '.onnx',
    'Config/NeuralNets/RobotDetector/' + MODEL + '.hdf5',
    'Util/onnxruntime/LICENSE', 'Util/onnxruntime/ThirdPartyNotices.txt',
    'Util/onnxruntime/VERSION_NUMBER', 'Util/onnxruntime/GIT_COMMIT_ID',
    'Util/onnxruntime/include/onnxruntime_c_api.h',
    'Util/onnxruntime/include/onnxruntime_cxx_api.h',
    'Util/onnxruntime/include/onnxruntime_cxx_inline.h',
)
BINARY_DEPS = (
    'Util/onnxruntime/lib/Windows/onnxruntime.dll',
    'Util/onnxruntime/lib/Windows/onnxruntime.lib',
    'Util/onnxruntime/lib/Linux/libonnxruntime.so.1.10.0',
)


def migrate(source):
    source = Path(source).resolve()
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError('expected the same immutable revision as the motion baseline')
    def blob(name):
        return subprocess.check_output(['git', '-C', str(source), 'show', COMMIT + ':' + name])
    manifest = {'version': 'bhuman_robot_originals_v1', 'source': UPSTREAM,
                'commit': COMMIT, 'files': {}, 'binary_dependencies': {}}
    for name in FILES:
        data = blob(name)
        path = DEST / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError('refusing to replace changed original: ' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest['files'][name] = hashlib.sha256(data).hexdigest()
    for name in BINARY_DEPS:
        data = blob(name)
        manifest['binary_dependencies'][name] = {
            'url': 'https://raw.githubusercontent.com/juanulloa2050/SabanaHerons2026/' + COMMIT + '/' + name,
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        cached = ROOT / 'native/bhuman_motion/build/vision_deps' / Path(name).name
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    manifest = migrate(args.source)
    print('Imported %d unchanged originals; motion baseline untouched.' % len(manifest['files']))
