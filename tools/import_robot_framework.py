"""Import the B-Human 2023 / SabanaHerons2026 framework subset that the NAO needs.

Copies, from the pinned SabanaHerons2026 revision (B-Human Code Release 2023
base), every file required to cross-compile the `bhuman` executable for a
NAO V6 and to deploy it over Ethernet: CMake, deploy/install scripts, sources,
configuration (without SimRobot scenes), the V6 sysroot and the Linux/V6
libraries. Windows/macOS binaries, SimRobot, Python bindings and desktop-only
libraries are left out. The CompiledNN and SimRobot (Eigen only) submodules are
taken from local checkouts of their pinned commits.

Every imported file is recorded with its SHA-256 in robot/PROVENANCE.json, so
`tools/robot_build.py --check-originals` can list which originals PenaltyStack
modified. Penalty-specific code lives in new files; see robot/README.md.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'robot'
COMMIT = '2360ab1100ad28f70e190c893f6b095651ff0bbc'
UPSTREAM = 'https://github.com/juanulloa2050/SabanaHerons2026'
SUBMODULES = {
    'Util/CompiledNN': ('https://github.com/bhuman/CompiledNN',
                        '091512e44520debafe3f502974ee4c0c39158794', None),
    'Util/SimRobot': ('https://github.com/bhuman/SimRobot',
                      '732794cf0f7d617f92bba6a37c284b6b4ceb7912',
                      ('Make/CMake/Eigen.cmake', 'Util/Eigen/')),
}
INCLUDE = (
    'CMakeLists.txt', 'CMakePresets.json', 'License.md', 'CITATION.cff', '.editorconfig',
    'Make/CMake/', 'Make/Common/', 'Make/Linux/',
    'Src/', 'Config/', 'Install/',
    'Util/Buildchain/V6/', 'Util/CompiledNN2ONNX/include/', 'Util/GameController/', 'Util/MD5/',
    'Util/fftw-3.3/COPYRIGHT', 'Util/fftw-3.3/VERSION.txt', 'Util/fftw-3.3/fftw3.h', 'Util/fftw-3.3/V6/',
    'Util/flite/COPYING', 'Util/flite/VERSION', 'Util/flite/include/', 'Util/flite/lib/V6/',
    'Util/hdf5/COPYING', 'Util/hdf5/VERSION', 'Util/hdf5/include/', 'Util/hdf5/lib/V6/',
    'Util/libjpeg/README', 'Util/libjpeg/VERSION', 'Util/libjpeg/include/', 'Util/libjpeg/lib/V6/',
    'Util/onnxruntime/LICENSE', 'Util/onnxruntime/ThirdPartyNotices.txt',
    'Util/onnxruntime/VERSION_NUMBER', 'Util/onnxruntime/GIT_COMMIT_ID',
    'Util/onnxruntime/include/', 'Util/onnxruntime/lib/Linux/',
    'Util/sse2neon/', 'Util/voronoi/',
)
EXCLUDE = ('Config/Scenes/',)


def _selected(name, include=INCLUDE, exclude=EXCLUDE):
    if any(name.startswith(prefix) for prefix in exclude):
        return False
    if include is None:
        return True
    return any(name == entry or (entry.endswith('/') and name.startswith(entry)) for entry in include)


def _record(manifest, name, data, mode, origin):
    path = DEST / name
    if path.exists() and path.read_bytes() != data:
        raise ValueError('refusing to replace a modified file: robot/' + name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    manifest['files'][name] = {'sha256': hashlib.sha256(data).hexdigest(),
                               'executable': bool(mode & 0o111), 'origin': origin}


def _normalize(path):
    parts = []
    for part in path.split('/'):
        if part == '..':
            parts.pop()
        elif part not in ('', '.'):
            parts.append(part)
    return '/'.join(parts)


def _read(tar, member):
    stream = tar.extractfile(member)
    if stream is None:
        raise ValueError('not a regular file in archive: ' + member.name)
    return stream.read()


def _import_archive(manifest, repository, commit, origin, prefix='', include=INCLUDE, exclude=EXCLUDE,
                    pathspecs=()):
    # git archive reads the object store at an exact commit, so sparse or partial
    # checkouts still export the full pinned revision.
    archive = subprocess.check_output(['git', '-C', str(Path(repository).resolve()), 'archive',
                                       '--format=tar', commit, '--', *pathspecs])
    symlinks = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = {m.name: m for m in tar.getmembers()}
        for name, member in members.items():
            if not _selected(name, include, exclude):
                continue
            if member.issym():
                symlinks[name] = member.linkname
            elif member.isfile():
                _record(manifest, prefix + name, _read(tar, member), member.mode, origin)
        # Windows checkouts cannot rely on symlinks: store the target's bytes instead.
        for name, target in symlinks.items():
            resolved = _normalize(str(Path(name).parent / target).replace('\\', '/'))
            member = members[resolved]
            _record(manifest, prefix + name, _read(tar, member), member.mode, origin)
            manifest['materialized_symlinks'][prefix + name] = prefix + resolved


def import_framework(source, compilednn, simrobot):
    manifest = {'version': 'penaltystack_robot_framework_v1', 'source': UPSTREAM, 'commit': COMMIT,
                'base': 'B-Human Code Release 2023', 'submodules': {}, 'materialized_symlinks': {},
                'files': {}}
    _import_archive(manifest, source, COMMIT, 'SabanaHerons2026')
    for prefix, checkout in (('Util/CompiledNN', compilednn), ('Util/SimRobot', simrobot)):
        url, commit, subset = SUBMODULES[prefix]
        manifest['submodules'][prefix] = {'url': url, 'commit': commit}
        _import_archive(manifest, checkout, commit, prefix, prefix + '/', subset, (), subset or ())
    (DEST / 'PROVENANCE.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n',
                                          encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import the NAO framework subset into robot/.')
    parser.add_argument('--source', required=True, help='SabanaHerons2026 checkout (any sparse state)')
    parser.add_argument('--compilednn', required=True, help='bhuman/CompiledNN checkout at the pinned commit')
    parser.add_argument('--simrobot', required=True, help='bhuman/SimRobot checkout with Eigen at the pinned commit')
    args = parser.parse_args()
    result = import_framework(args.source, args.compilednn, args.simrobot)
    total = sum(1 for _ in result['files'])
    print('Imported %d framework files into robot/ (%d materialized symlinks).'
          % (total, len(result['materialized_symlinks'])))
