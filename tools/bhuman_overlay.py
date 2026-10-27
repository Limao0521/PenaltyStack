"""Apply PenaltyStack's B-Human files onto a clean SabanaHerons2026 checkout.

PenaltyStack owns every B-Human source and config file the penalty test needs
(``bhuman/files``). They are pinned to one commit of the team fork
(``bhuman/overlay.json``). This tool copies them onto a dedicated checkout of
that commit (a git worktree), so the fork itself never receives commits and a
developer's own work in another checkout is never touched.

    python tools/bhuman_overlay.py worktree   # create the clean worktree once
    python tools/bhuman_overlay.py apply      # copy overlay files into it
    python tools/bhuman_overlay.py status     # which files differ
    python tools/bhuman_overlay.py build      # Nao (and DeployDialog) build
    python tools/bhuman_overlay.py capture PATH...  # take files from the worktree back
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bhuman_cfg import check_tree  # noqa: E402
OVERLAY = ROOT / "bhuman"
FILES = OVERLAY / "files"
MANIFEST = OVERLAY / "overlay.json"
STATE_NAME = "penaltystack-overlay.json"  # stored under <target>/Build


class OverlayError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git(target: Path, *args: str, binary: bool = False):
    result = subprocess.run(["git", "-C", str(target), *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise OverlayError(f"git {' '.join(args)}: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout if binary else result.stdout.decode().strip()


def base_blob(target: Path, commit: str, path: str) -> bytes | None:
    result = subprocess.run(["git", "-C", str(target), "show", f"{commit}:{path}"],
                            capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def state_path(target: Path) -> Path:
    return target / "Build" / STATE_NAME


def read_state(target: Path) -> dict:
    path = state_path(target)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": {}}


def plan(target: Path, manifest: dict, files: Path = FILES) -> tuple[list[str], list[str], list[str]]:
    """Return (to_write, to_restore, conflicts) for applying the overlay to target."""
    commit = manifest["base_commit"]
    head = git(target, "rev-parse", "HEAD")
    if head != commit:
        raise OverlayError(f"{target} is at {head[:10]}, overlay needs {commit[:10]} (use a clean worktree)")
    previous = read_state(target)["files"]
    to_write, conflicts = [], []
    for rel, entry in manifest["files"].items():
        wanted = (files / rel).read_bytes()
        current_path = target / rel
        current = current_path.read_bytes() if current_path.exists() else None
        if current == wanted:
            continue
        base = base_blob(target, commit, rel) if entry["kind"] == "modified" else None
        if current is None or current == base or previous.get(rel) == (sha256(current) if current else None):
            to_write.append(rel)
        else:
            conflicts.append(rel)
    to_restore = [rel for rel in previous if rel not in manifest["files"]]
    return to_write, to_restore, conflicts


def apply(target: Path, manifest: dict, files: Path = FILES) -> list[str]:
    broken = check_tree(files)
    if broken:
        raise OverlayError("B-Human would reject these .cfg files (red eyes):\n  " + "\n  ".join(broken))
    to_write, to_restore, conflicts = plan(target, manifest, files)
    if conflicts:
        raise OverlayError("changed outside the overlay (not overwritten): " + ", ".join(conflicts))
    commit = manifest["base_commit"]
    for rel in to_restore:
        base = base_blob(target, commit, rel)
        if base is None:
            (target / rel).unlink(missing_ok=True)
        else:
            (target / rel).write_bytes(base)
    for rel in to_write:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Fresh mtime on purpose: ninja must rebuild objects older than the overlay.
        shutil.copyfile(files / rel, dst)
    state = {"base_commit": commit,
             "files": {rel: sha256((files / rel).read_bytes()) for rel in manifest["files"]}}
    state_path(target).parent.mkdir(parents=True, exist_ok=True)
    state_path(target).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return to_write + to_restore


def capture(target: Path, paths: list[str], manifest: dict, files: Path = FILES) -> dict:
    """Copy files from target into the overlay and record whether the base had them."""
    commit = manifest["base_commit"]
    for rel in paths:
        src = target / rel
        if not src.is_file():
            raise OverlayError(f"{rel} does not exist in {target}")
        dst = files / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        kind = "modified" if base_blob(target, commit, rel) is not None else "added"
        manifest["files"][rel] = {"kind": kind}
    manifest["files"] = dict(sorted(manifest["files"].items()))
    # The target now holds exactly the captured content: record it as overlay-owned,
    # so the next apply may replace it again.
    if state_path(target).exists():
        state = read_state(target)
        for rel in paths:
            state["files"][rel] = sha256((target / rel).read_bytes())
        state_path(target).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def save_manifest(manifest: dict, path: Path = MANIFEST) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def ensure_worktree(target: Path, manifest: dict) -> None:
    source = Path(manifest["source_checkout"]).expanduser()
    if not target.exists():
        git(source, "worktree", "add", "--detach", str(target), manifest["base_commit"])
    for sub in manifest.get("submodules", []):
        if not any((target / sub).iterdir()) if (target / sub).exists() else True:
            subprocess.run(["git", "-C", str(target), "-c", "protocol.file.allow=always", "submodule", "update",
                            "--init", "--reference", str(source / sub), "--", sub], check=True)


def build(target: Path, manifest: dict, targets: list[str]) -> None:
    build_dir = target / "Build/Linux/CMake/Develop"
    if not (build_dir / "build.ninja").exists():
        tools = manifest.get("toolchain", {})
        cmd = ["cmake", "-DCMAKE_BUILD_TYPE=Develop", "-G", "Ninja", "-S", str(target / "Make/Linux"),
               "-B", str(build_dir)]
        cmd += [f"-D{key}={value}" for key, value in tools.items()]
        subprocess.run(cmd, check=True)
    for name in targets:
        subprocess.run(["cmake", "--build", str(build_dir), "--target", name, "-j", "12"], check=True)


def main(argv: list[str] | None = None) -> int:
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=Path, default=Path(manifest["worktree"]).expanduser(),
                        help="clean SabanaHerons2026 checkout at the base commit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("worktree")
    sub.add_parser("apply")
    sub.add_parser("status")
    b = sub.add_parser("build")
    b.add_argument("--no-deploy-dialog", action="store_true")
    c = sub.add_parser("capture")
    c.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    target = args.target.expanduser().resolve()
    try:
        if args.command == "worktree":
            ensure_worktree(target, manifest)
            print(f"worktree ready: {target}")
        elif args.command == "apply":
            changed = apply(target, manifest)
            print(f"{len(changed)} file(s) updated in {target}")
            for rel in changed:
                print(f"  {rel}")
        elif args.command == "status":
            to_write, to_restore, conflicts = plan(target, manifest)
            print(f"base {manifest['base_commit'][:10]}  overlay files {len(manifest['files'])}")
            for label, items in (("pending", to_write), ("to restore", to_restore), ("conflict", conflicts)):
                for rel in items:
                    print(f"  {label}: {rel}")
            if not (to_write or to_restore or conflicts):
                print("  applied, nothing pending")
        elif args.command == "build":
            apply(target, manifest)
            build(target, manifest, ["Nao"] + ([] if args.no_deploy_dialog else ["DeployDialog"]))
        elif args.command == "capture":
            save_manifest(capture(target, args.paths, manifest))
            print(f"captured {len(args.paths)} file(s)")
    except OverlayError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
