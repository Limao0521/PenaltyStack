"""Build the isolated, original B-Human motion subsystem with MSVC/CMake.

Original baseline files are never edited. The only generated source patch
removes the Sabana POSIX RL diagnostic side channel, unused by this runtime.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import urllib.request
import zipfile
import io

ROOT = pathlib.Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'native' / 'bhuman_motion'
DEPS = NATIVE / 'dependencies'


def dependency(name, url, expected=None):
    path = DEPS / name
    if path.exists():
        return path
    data = urllib.request.urlopen(url, timeout=90).read()
    digest = hashlib.sha256(data).hexdigest()
    if expected and digest != expected:
        raise ValueError('dependency checksum differs: ' + name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def prepare():
    manifest = json.loads((ROOT / 'baseline/motion_manifest.json').read_text(encoding='utf-8'))
    if manifest['commit'] != '2360ab1100ad28f70e190c893f6b095651ff0bbc':
        raise ValueError('B-Human baseline is not the configured source revision')
    for name, expected_hash in manifest['files'].items():
        path = ROOT / 'baseline/bhuman' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Original baseline modified: ' + name)
    lock_path = NATIVE / 'dependencies.lock.json'
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    urls = {
        'eigen-3.4.0.zip': 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip',
        'json.hpp': 'https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp',
        'RoboCupGameControlData.h': 'https://raw.githubusercontent.com/RoboCup-HumanoidSoccerLeague/GameController/81f0e79d8c84a9b24db47a8e7feaf2a3486bdb60/game_controller_msgs/headers/RoboCupGameControlData.h',
    }
    for name, url in urls.items():
        path = dependency(name, url, lock.get(name, {}).get('sha256'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in lock and lock[name]['sha256'] != digest:
            raise ValueError('cached dependency checksum differs: ' + name)
        lock[name] = {'url': url, 'sha256': digest}
    lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    eigen = DEPS / 'eigen-3.4.0'
    if not eigen.exists():
        with zipfile.ZipFile(io.BytesIO((DEPS / 'eigen-3.4.0.zip').read_bytes())) as archive:
            for member in archive.infolist():
                destination = (DEPS / member.filename).resolve()
                if not destination.is_relative_to(DEPS.resolve()):
                    raise ValueError('archive member escapes dependency directory')
            archive.extractall(DEPS)
    original = ROOT / 'baseline/bhuman/Src/Modules/MotionControl/MotionEngine/MotionEngine.cpp'
    source = original.read_text(encoding='utf-8')
    source = source.replace('#include "Python/Controller/RLSharedState.h"', '')
    start = source.index('  if(RLSharedStateBridge::isEnabledForTeam(')
    end = source.index('  // Construct odometry update', start)
    source = source[:start] + source[end:]
    generated = NATIVE / 'build' / 'generated'
    generated.mkdir(parents=True, exist_ok=True)
    config_hashes = {name: value for name, value in manifest['files'].items() if name.startswith('Config/')}
    provenance = {'source_commit': manifest['commit'],
                  'manifest_sha256': hashlib.sha256((ROOT / 'baseline/motion_manifest.json').read_bytes()).hexdigest(),
                  'configuration_sha256': hashlib.sha256(json.dumps(config_hashes,sort_keys=True).encode()).hexdigest(),
                  'original_file_count': len(manifest['files']),
                  'inputs_profile': 'mujoco-truth-v2',
                  'simulation_calibration': {'ball_radius':'INIT geometry in mm',
                      'inertial_angle':'original removeZRotation and angle-axis',
                      'right_forward_stroke_ms':200,'left_forward_stroke_ms':100},
                  'host_source_sha256': hashlib.sha256(b''.join(
                      path.relative_to(ROOT).as_posix().encode()+b'\0'+path.read_bytes()
                      for path in sorted(list(NATIVE.glob('*.cpp'))+list(NATIVE.glob('*.h'))+
                                         [NATIVE/'CMakeLists.txt',pathlib.Path(__file__).resolve()]))).hexdigest()}
    (generated / 'provenance.h').write_text('static constexpr const char* provenance_json = R"PROV(' +
                                          json.dumps(provenance) + ')PROV";\n', encoding='utf-8')
    (generated / 'MotionEngine.cpp').write_text(source, encoding='utf-8')
    file_source = (ROOT / 'baseline/bhuman/Src/Libs/Platform/File.cpp').read_text(encoding='utf-8')
    start = file_source.index('const char* File::getBHDir()')
    end = file_source.index('bool File::isAbsolute(', start)
    file_source = file_source[:start] + 'const char* File::getBHDir() { return BHUMAN_BASELINE_DIR; }\n\n' + file_source[end:]
    file_source = file_source.replace('#include "File.h"', '#include "Platform/File.h"')
    (generated / 'File.cpp').write_text(file_source, encoding='utf-8')
    pose_source = (ROOT / 'baseline/bhuman/Src/Modules/MotionControl/WalkingEngine/WalkGenerators/WalkToPoseEngine.cpp').read_text(encoding='utf-8')
    pose_source = pose_source.replace('#include "Python/Controller/RLSharedState.h"', '')
    start = pose_source.index('  if(RLSharedStateBridge::isEnabledForTeam(')
    opening = pose_source.index('{', start)
    level, end = 1, opening + 1
    while level:
        if pose_source[end] == '{': level += 1
        if pose_source[end] == '}': level -= 1
        end += 1
    pose_source = pose_source[:start] + pose_source[end:]
    (generated / 'WalkToPoseEngine.cpp').write_text(pose_source, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-only', action='store_true')
    args = parser.parse_args()
    prepare()
    vswhere = pathlib.Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
    vs = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
                                 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                                 '-property', 'installationPath'], text=True).strip()
    if not vs:
        raise RuntimeError('Install Visual Studio 2022 Build Tools: Desktop development with C++')
    cmake = shutil.which('cmake') or str(pathlib.Path(vs) / 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe')
    subprocess.run([cmake, '-S', str(NATIVE), '-B', str(NATIVE / 'build'), '-G',
                    'Visual Studio 17 2022', '-A', 'x64', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'], check=True)
    if not args.configure_only:
        subprocess.run([cmake, '--build', str(NATIVE / 'build'), '--config', 'Release',
                        '--parallel', '4'], check=True)
        print(NATIVE / 'build/Release/bhuman_motion.exe')


if __name__ == '__main__':
    main()
