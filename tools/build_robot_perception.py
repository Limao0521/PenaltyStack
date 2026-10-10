"""Build a local, sensor-only host of the original B-Human robot detector core.

Selected original preprocessing, box decoding and suppression bodies are
compiled verbatim. Framework/platform declarations are generated separately.
No SimRobot, game controller, LoLA, motions or network socket is linked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'heritage/bhuman_robot_perception'
BUILD = ROOT / 'native/bhuman_motion/build'
GENERATED = BUILD / 'generated/vision'


def function(source, signature):
    start = source.index(signature)
    opening = source.index('{', start)
    level, end = 1, opening + 1
    while level:
        if source[end] == '{':
            level += 1
        elif source[end] == '}':
            level -= 1
        end += 1
    return source[start:end]


def prepare():
    manifest = json.loads((ORIGINAL / 'manifest.json').read_text())
    motion = json.loads((ROOT / 'baseline/motion_manifest.json').read_text())
    if manifest['commit'] != motion['commit']:
        raise ValueError('vision/motion source revisions differ')
    for base, files in ((ORIGINAL, manifest['files']), (ROOT / 'baseline/bhuman', motion['files'])):
        for name, expected in files.items():
            if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
                raise ValueError('immutable original changed: ' + name)
    GENERATED.mkdir(parents=True, exist_ok=True)
    deps = BUILD / 'vision_deps'
    deps.mkdir(parents=True, exist_ok=True)
    for name, info in manifest['binary_dependencies'].items():
        # Linux artifacts are downloaded only when building a Linux host.
        if ('/Windows/' in name) != (os.name == 'nt'):
            continue
        target = deps / Path(name).name
        if not target.exists():
            data = urllib.request.urlopen(info['url'], timeout=90).read()
            if hashlib.sha256(data).hexdigest() != info['sha256']:
                raise ValueError('downloaded runtime checksum differs')
            target.write_bytes(data)
        if hashlib.sha256(target.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('cached runtime checksum differs')
    eigen = ROOT / 'native/bhuman_motion/dependencies/eigen-3.4.0'
    json_header = eigen.parent / 'json.hpp'
    if not eigen.is_dir() or not json_header.is_file():
        from build_motion_runtime import prepare as prepare_motion_dependencies
        prepare_motion_dependencies()
    cfg = (ORIGINAL / 'Config/NeuralNets/RobotDetector/4_anchor_boxes_20230629-220730.cfg').read_text()
    values = dict(re.findall(r'(\w+)\s*=\s*([\d.]+);', cfg))
    anchors = re.findall(r'\{\s*x\s*=\s*([\d.]+);\s*y\s*=\s*([\d.]+);\s*\}', cfg)
    config = 'struct NetworkParameters {\n'
    for name in ('inputHeight', 'inputWidth', 'inputChannels', 'outputHeight', 'outputWidth',
                 'outputAnchors', 'paramsPerAnchor', 'confidenceIndex', 'yMidIndex', 'xMidIndex', 'heightIndex', 'widthIndex'):
        config += '  unsigned ' + name + ' = ' + values[name] + ';\n'
    config += '  float sizeConversionFactor = ' + values['sizeConversionFactor'] + 'f;\n'
    config += '  std::vector<Vector2f> anchors = {' + ','.join('Vector2f(%sf,%sf)' % a for a in anchors) + '};\n};\n'
    (GENERATED / 'parameters.h').write_text(config, encoding='utf-8')
    detector = (ORIGINAL / 'Src/Modules/Perception/PlayersPerceptors/RobotDetector.cpp').read_text()
    resize = (ORIGINAL / 'Src/Libs/ImageProcessing/Resize.cpp').read_text()
    labels = (ORIGINAL / 'Src/Libs/ImageProcessing/LabelImage.cpp').read_text()
    conversion = (ORIGINAL / 'Src/Libs/ImageProcessing/ColorModelConversions.h').read_text()
    conversion = conversion[conversion.index('  static constexpr float yCoeffR'):conversion.index('  /**\n   * Converts an RGB pixel into an HSI pixel.')]
    parts = [function(resize, 'void Resize::shrinkY(')]
    parts += [function(labels, s) for s in ('bool LabelImage::Annotation::isInside(',
              'bool LabelImage::Annotation::operator<(', 'float LabelImage::Annotation::getArea(',
              'float LabelImage::Annotation::getOverlap(', 'float LabelImage::Annotation::getIou(',
              'void LabelImage::nonMaximumSuppression(', 'void LabelImage::bigBoxSuppression(')]
    parts += [function(detector, s) for s in ('void RobotDetector::fillGrayscaleThumbnail(',
              'void RobotDetector::fillUVThumbnails(', 'void RobotDetector::applyColorNetwork(',
              'template<typename ConvModel>\nvoid RobotDetector::boundingBoxes(',
              'LabelImage::Annotation RobotDetector::predictionToBoundingBox(')]
    parts += [function(detector, s) for s in ('void RobotDetector::scanImage(',
              'void RobotDetector::classifyRegions(', 'void RobotDetector::discardHomogeneousAreas(',
              'void RobotDetector::dbScan(', 'bool RobotDetector::expandCluster(',
              'void RobotDetector::regionQuery(')]
    algorithms = 'namespace ColorModelConversions {\n' + conversion + '}\n\n' + '\n\n'.join(parts) + '\n'
    (GENERATED / 'original_algorithms.h').write_bytes(algorithms.encode('utf-8'))
    provenance = {'protocol': 'bhuman_robot_pixels_v1', 'source_commit': manifest['commit'],
                  'onnxruntime': '1.10.0', 'algorithm_bodies': 'original C++ bodies, unmodified',
                  'original_manifest_sha256': hashlib.sha256((ORIGINAL / 'manifest.json').read_bytes()).hexdigest(),
                  'generated_algorithms_sha256': hashlib.sha256(algorithms.encode()).hexdigest(),
                  'host_source_sha256': hashlib.sha256((ROOT / 'native/bhuman_vision/main.cpp').read_bytes()).hexdigest(),
                  'model_sha256': manifest['files']['Config/NeuralNets/RobotDetector/4_anchor_boxes_model_no_activation_20230629-220730.onnx'],
                  'component': 'RobotDetector upper network/preprocessing/decoding/NMS plus original lower contrast/clustering',
                  'lower_trim_obstacles': False,
                  'lower_saturation_adapter': 'scalar lighting-independent saturation; not bit-identical to original JIT approximations',
                  'not_executed': ['trimObstacle', 'JerseyClassifier', 'ObstacleModelProvider', 'body/field boundary providers'],
                  'actuators': False}
    (GENERATED / 'provenance.h').write_text('static constexpr const char* provenance_json = R"PROV(' + json.dumps(provenance) + ')PROV";\n')
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-only', action='store_true')
    args = parser.parse_args()
    prepare()
    native = ROOT / 'native/bhuman_vision'
    output = BUILD / 'vision'
    cmake = shutil.which('cmake')
    configure = []
    if os.name == 'nt':
        vswhere = Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
        vs = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
                                      'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-format', 'json'], text=True)
        installations = json.loads(vs)
        if not installations:
            raise RuntimeError('MSVC Build Tools with C++ are required')
        install = installations[0]
        version = int(install['installationVersion'].split('.')[0])
        generator = {17: 'Visual Studio 17 2022', 18: 'Visual Studio 18 2026'}[version]
        cmake = cmake or str(Path(install['installationPath']) / 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe')
        configure = ['-G', generator, '-A', 'x64']
    if not cmake:
        raise RuntimeError('CMake is required')
    subprocess.run([cmake, '-S', str(native), '-B', str(output), *configure,
                    '-DCMAKE_BUILD_TYPE=Release'], check=True)
    if not args.configure_only:
        subprocess.run([cmake, '--build', str(output), '--config', 'Release', '--parallel', '4'], check=True)
        print(output / ('Release/bhuman_vision.exe' if os.name == 'nt' else 'bhuman_vision'))


if __name__ == '__main__':
    main()
