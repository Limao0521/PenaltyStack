"""Cross-compile the NAO V6 `bhuman` executable from robot/ (B-Human 2023 framework).

On Windows every step runs inside WSL: robot/ is mirrored to the WSL ext4
filesystem (much faster than /mnt/c), configured with the original
NaoToolchain.cmake (clang + mold + V6 sysroot) and built with Ninja. On Linux
the same script runs natively. The result stays in the WSL work tree, from
where deploy/robot_deploy.py copies it to the robots.

    python tools/robot_build.py                     # Release build
    python tools/robot_build.py --config Develop
    python tools/robot_build.py --install-toolchain # clang/mold/cmake without sudo
    python tools/robot_build.py --check             # report toolchain only
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / 'robot'
CONFIGS = ('Release', 'Develop', 'Debug')
DEFAULT_DISTRO = os.environ.get('PENALTYSTACK_WSL_DISTRO', 'Ubuntu-24.04')
WORK = '$HOME/.cache/penaltystack/robot'
TOOLCHAIN = '$HOME/.local/penaltystack-toolchain'
# Ubuntu 24.04 packages for a user-level toolchain (no sudo): clang 18, mold, cmake, ccache.
TOOLCHAIN_PACKAGES = ('ccache libhiredis1.1.0 clang-18 cmake cmake-data libarchive13t64 libclang-common-18-dev libclang-cpp18 '
                      'libclang1-18 libgc1 libjsoncpp25 libllvm18 libobjc-13-dev libobjc4 librhash0 '
                      'llvm-18-linker-tools mold')
# Scripts added by PenaltyStack (the originals' executable bits come from PROVENANCE.json).
EXTRA_EXECUTABLES = ('Make/Common/deployCopaPenalty',)

ENVIRONMENT = r'''
TC="%(toolchain)s"
if [ -x "$TC/bin/clang++" ]; then
  export PATH="$TC/bin:$PATH"
  export LD_LIBRARY_PATH="$TC/root/usr/lib/x86_64-linux-gnu:$TC/root/usr/lib/llvm-18/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
''' % {'toolchain': TOOLCHAIN}


def to_wsl_path(path):
    path = Path(path).resolve()
    if os.name != 'nt':
        return str(path)
    drive, rest = path.drive, path.as_posix()[len(path.drive):]
    return '/mnt/' + drive[0].lower() + rest


def run_bash(script, distro=DEFAULT_DISTRO, check=True, capture=False):
    """Run a bash script natively or in WSL, streaming output unless captured."""
    if os.name == 'nt':
        # --exec skips WSL's outer shell, which would expand $variables early.
        command = ['wsl.exe', '-d', distro, '--exec', 'bash', '-lc', script]
    else:
        command = ['bash', '-lc', script]
    env = dict(os.environ, MSYS_NO_PATHCONV='1')
    if capture:
        return subprocess.run(command, env=env, check=check, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return subprocess.run(command, env=env, check=check, text=True)


def executables():
    provenance = json.loads((ROBOT / 'PROVENANCE.json').read_text(encoding='utf-8'))
    names = [name for name, info in provenance['files'].items() if info.get('executable')]
    return sorted(set(names) | set(EXTRA_EXECUTABLES))


def modified_originals():
    """Imported originals whose bytes changed, and files added by PenaltyStack (robot/ only)."""
    import hashlib
    provenance = json.loads((ROBOT / 'PROVENANCE.json').read_text(encoding='utf-8'))
    changed, missing = [], []
    for name, info in sorted(provenance['files'].items()):
        path = ROBOT / name
        if not path.is_file():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            changed.append(name)
    added = sorted(p.relative_to(ROBOT).as_posix() for p in ROBOT.rglob('*')
                   if p.is_file() and not p.relative_to(ROBOT).as_posix().startswith('Build/')
                   and p.relative_to(ROBOT).as_posix() not in provenance['files']
                   and p.name != 'PROVENANCE.json')
    return changed, missing, added


def sync_script():
    """Mirror robot/ into the WSL work tree and restore executable bits."""
    chmod = ' '.join(shlex.quote(name) for name in executables())
    return ENVIRONMENT + r'''
set -euo pipefail
SRC=%(src)s
WORK="%(work)s"
mkdir -p "$WORK"
rsync -a --delete --exclude=/Build/ --exclude=/PROVENANCE.json "$SRC/" "$WORK/"
cd "$WORK"
for f in %(chmod)s; do [ -f "$f" ] && chmod +x "$f"; done
''' % {'src': shlex.quote(to_wsl_path(ROBOT)), 'work': WORK, 'chmod': chmod}


def check_script():
    return ENVIRONMENT + r'''
missing=0
for tool in clang++ ld.mold cmake ccache ninja rsync ssh; do
  if command -v $tool >/dev/null 2>&1; then echo "ok      $tool -> $(command -v $tool)"; else echo "missing $tool"; missing=1; fi
done
exit $missing
'''


def build_script(config, jobs):
    return sync_script() + r'''
BUILD="$WORK/Build/Linux/CMake/%(config)s"
if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -S "$WORK/Make/Linux" -B "$BUILD" -G Ninja -DBUILD_NAO=ON -DCMAKE_BUILD_TYPE=%(config)s
fi
cmake --build "$BUILD" --target Nao -- -j%(jobs)s
BIN="$WORK/Build/Linux/Nao/%(config)s/bhuman"
test -x "$BIN"
echo "BHUMAN_BINARY=$BIN"
sha256sum "$BIN"
''' % {'config': config, 'jobs': jobs}


def install_toolchain_script():
    return r'''
set -euo pipefail
T="%(toolchain)s"
mkdir -p "$T/debs" "$T/root" "$T/bin"
cd "$T/debs"
apt-get download %(packages)s
for d in *.deb; do dpkg -x "$d" "$T/root"; done
ln -sf ../root/usr/lib/llvm-18/bin/clang "$T/bin/clang"
ln -sf ../root/usr/lib/llvm-18/bin/clang++ "$T/bin/clang++"
ln -sf ../root/usr/bin/cmake "$T/bin/cmake"
ln -sf ../root/usr/bin/mold "$T/bin/mold"
ln -sf ../root/usr/bin/mold "$T/bin/ld.mold"
ln -sf ../root/usr/bin/ccache "$T/bin/ccache"
echo "Toolchain installed in $T (ninja, rsync and ssh come from the distribution)."
''' % {'toolchain': TOOLCHAIN, 'packages': TOOLCHAIN_PACKAGES}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Cross-compile bhuman for NAO V6 (via WSL on Windows).')
    parser.add_argument('--config', choices=CONFIGS, default='Release')
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    parser.add_argument('--distro', default=DEFAULT_DISTRO, help='WSL distribution (Windows only)')
    parser.add_argument('--check', action='store_true', help='only report the toolchain')
    parser.add_argument('--sync-only', action='store_true', help='mirror robot/ into the work tree')
    parser.add_argument('--check-originals', action='store_true',
                        help='list modified B-Human/Sabana Herons originals and files added by PenaltyStack')
    parser.add_argument('--install-toolchain', action='store_true',
                        help='download clang/mold/cmake from the Ubuntu archive into the user profile')
    args = parser.parse_args(argv)
    if args.check_originals:
        changed, missing, added = modified_originals()
        for title, names in (('modified originals', changed), ('missing originals', missing), ('added files', added)):
            print('%s (%d):' % (title, len(names)))
            for name in names:
                print('  ' + name)
        return 1 if missing else 0
    if args.install_toolchain:
        return run_bash(install_toolchain_script(), args.distro, check=False).returncode
    if args.check:
        return run_bash(check_script(), args.distro, check=False).returncode
    if args.sync_only:
        return run_bash(sync_script(), args.distro, check=False).returncode
    if run_bash(check_script(), args.distro, check=False, capture=True).returncode != 0:
        print('Toolchain incompleto. Ejecuta: python tools/robot_build.py --install-toolchain '
              '(o en WSL: sudo apt install clang mold cmake ccache ninja-build rsync)', file=sys.stderr)
        return 2
    return run_bash(build_script(args.config, max(1, args.jobs)), args.distro, check=False).returncode


if __name__ == '__main__':
    sys.exit(main())
