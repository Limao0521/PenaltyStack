"""Checks of the B-Human overlay that would otherwise only fail on the robot (red eyes)."""

import importlib.util
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "bhuman/files"
COPA = ("CopaPenaltyStriker", "CopaPenaltyKeeper")

spec = importlib.util.spec_from_file_location("bhuman_overlay", ROOT / "tools/bhuman_overlay.py")
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


def module_header(name: str) -> str:
    return next(FILES.joinpath("Src/Modules").rglob(f"{name}.h")).read_text()


def loaded_parameters(name: str) -> set[str]:
    block = module_header(name).split("LOADS_PARAMETERS(", 1)[1].split("}),", 1)[0]
    return set(re.findall(r"^\s*\([^)]*\)(?:\([^)]*\))?\s*(\w+),", block, re.M))


def cfg_keys(path: Path) -> set[str]:
    text = re.sub(r"//[^\n]*", "", path.read_text())
    return set(re.findall(r"^\s*(\w+)\s*=", text, re.M))


def requires(name: str) -> set[str]:
    return set(re.findall(r"REQUIRES\((\w+)\)", module_header(name)))


def thread_providers(scenario: str) -> dict[str, dict[str, str]]:
    text = (FILES / "Config/Scenarios" / scenario / "threads.cfg").read_text()
    threads = {}
    for name, body in re.findall(r"name = (\w+);(.*?)\]\s*;", text, re.S):
        threads[name] = dict(re.findall(r"\{representation = (\w+); provider = (\w+);\}", body))
    return threads


class CopaConfigTests(unittest.TestCase):
    def test_module_parameters_are_complete(self):
        for scenario in COPA:
            folder = FILES / "Config/Scenarios" / scenario
            for module, cfg in (("YoloBallDetector", "yoloBallDetector.cfg"),
                                ("CopaVisionPerceptor", "copaVisionPerceptor.cfg"),
                                ("CopaCameraStreamer", "cameraStreamer.cfg")):
                with self.subTest(scenario=scenario, module=module):
                    missing = loaded_parameters(module) - cfg_keys(folder / cfg)
                    self.assertFalse(missing, f"{cfg} lacks {sorted(missing)}")

    def test_every_overlay_yolo_cfg_has_the_detector_keys(self):
        keys = loaded_parameters("YoloBallDetector")
        for path in FILES.joinpath("Config").rglob("yoloBallDetector.cfg"):
            with self.subTest(path=str(path.relative_to(FILES))):
                self.assertFalse(keys - cfg_keys(path))

    def test_copa_modules_get_their_inputs_in_both_camera_threads(self):
        for scenario in COPA:
            threads = thread_providers(scenario)
            for thread in ("Upper", "Lower"):
                providers = threads[thread]
                self.assertEqual(providers["CameraStream"], "CopaCameraStreamer")
                self.assertEqual(providers["CopaVisionPercept"], "CopaVisionPerceptor")
                for module in ("CopaVisionPerceptor", "CopaCameraStreamer"):
                    missing = requires(module) - providers.keys()
                    with self.subTest(scenario=scenario, thread=thread, module=module):
                        self.assertFalse(missing)

    def test_copa_scenarios_pin_both_ball_models(self):
        models = FILES / "Config/NeuralNets/BallDetector"
        header = (FILES / "Src/Modules/Perception/BallPerceptors/CopaBallModel.h").read_text()
        for name in re.findall(r'"NeuralNets/BallDetector/(\w+\.onnx)"', header):
            self.assertTrue((models / name).is_file(), name)

    def test_team_uses_copa_location_and_never_dives(self):
        teams = (FILES / "Config/teams.cfg").read_text()
        self.assertIn("location = CopaPenalty;", teams)
        self.assertIn("goalkeeperDivingEnabled = false;", teams)
        ball = (FILES / "Config/Locations/CopaPenalty/ballSpecification.cfg").read_text()
        self.assertRegex(ball, r"radius = 32\.5;")


class OverlayToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.repo = base / "fork"
        self.files = base / "files"
        self.repo.mkdir()
        run = lambda *a: subprocess.run(["git", "-C", str(self.repo), *a], check=True, capture_output=True)
        run("init", "-q")
        run("config", "user.email", "t@t")
        run("config", "user.name", "t")
        (self.repo / "a.cfg").write_text("base\n")
        run("add", ".")
        run("commit", "-qm", "base")
        self.commit = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
        (self.files).mkdir()
        (self.files / "a.cfg").write_text("overlay\n")
        (self.files / "new.h").write_text("added\n")
        self.manifest = {"base_commit": self.commit,
                         "files": {"a.cfg": {"kind": "modified"}, "new.h": {"kind": "added"}}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_apply_writes_and_reapply_is_clean(self):
        changed = overlay.apply(self.repo, self.manifest, self.files)
        self.assertEqual(sorted(changed), ["a.cfg", "new.h"])
        self.assertEqual((self.repo / "a.cfg").read_text(), "overlay\n")
        self.assertEqual(overlay.plan(self.repo, self.manifest, self.files), ([], [], []))

    def test_outside_edits_are_not_overwritten(self):
        (self.repo / "a.cfg").write_text("someone else\n")
        with self.assertRaises(overlay.OverlayError):
            overlay.apply(self.repo, self.manifest, self.files)
        self.assertEqual((self.repo / "a.cfg").read_text(), "someone else\n")

    def test_updated_overlay_replaces_its_own_previous_version(self):
        overlay.apply(self.repo, self.manifest, self.files)
        (self.files / "a.cfg").write_text("overlay v2\n")
        overlay.apply(self.repo, self.manifest, self.files)
        self.assertEqual((self.repo / "a.cfg").read_text(), "overlay v2\n")

    def test_files_dropped_from_overlay_are_restored(self):
        overlay.apply(self.repo, self.manifest, self.files)
        del self.manifest["files"]["a.cfg"]
        del self.manifest["files"]["new.h"]
        overlay.apply(self.repo, self.manifest, self.files)
        self.assertEqual((self.repo / "a.cfg").read_text(), "base\n")
        self.assertFalse((self.repo / "new.h").exists())

    def test_wrong_base_commit_is_refused(self):
        self.manifest["base_commit"] = "0" * 40
        with self.assertRaises(overlay.OverlayError):
            overlay.plan(self.repo, self.manifest, self.files)


if __name__ == "__main__":
    unittest.main()
