"""Provenance recorded with every evaluation, optimization or training run."""
import hashlib
import platform
import subprocess
import sys

from penalty.motion_contract import ROOT, SOURCE_COMMIT
from penalty.rl.contracts import versions

HASHED = ('penalty/rl/contracts.py', 'penalty/rl/rewards.py', 'penalty/rl/synthetic.py',
          'penalty/rl/bhuman_world.py', 'penalty/rl/observation.py', 'penalty/control.py',
          'penalty/physical_sim.py', 'penalty/sim_session.py', 'sim/scene.xml',
          'baseline/motion_manifest.json')


def _git(*args):
    try:
        return subprocess.run(('git',) + args, cwd=ROOT, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_metadata(**fields):
    """fields: seed(s), reward, opponent, model, backend, ... (JSON values)."""
    hashes = {}
    for name in HASHED:
        path = ROOT/name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    status = _git('status', '--porcelain', '--untracked-files=no')
    return dict(fields, commit=_git('rev-parse', 'HEAD'), dirty=bool(status) if status is not None else None,
                bhuman_source_commit=SOURCE_COMMIT, versions=versions(), hashes=hashes,
                python=sys.version.split()[0], platform=platform.platform())
