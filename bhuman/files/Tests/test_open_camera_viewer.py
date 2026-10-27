import os
import pathlib
import subprocess
import tempfile
import time
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "Make/Common/openCameraViewer"


class OpenCameraViewerTest(unittest.TestCase):
    def test_opens_dual_view_for_selected_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "args"
            fake = pathlib.Path(directory) / "python"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAMERA_TEST_OUTPUT"\n')
            fake.chmod(0o755)
            viewer = pathlib.Path(directory) / "viewer.py"
            viewer.touch()
            env = dict(os.environ, NAO_CAMERA_PYTHON=str(fake), NAO_CAMERA_VIEWER=str(viewer),
                       CAMERA_TEST_OUTPUT=str(output), DISPLAY=":test")
            subprocess.run([str(SCRIPT), "192.168.49.7"], env=env, check=True)
            for _ in range(100):
                if output.exists():
                    break
                time.sleep(.01)
            self.assertEqual(output.read_text().splitlines(),
                             [str(viewer), "--ip", "192.168.49.7", "--camera", "dual", "--no-monitor"])

    def test_missing_viewer_fails(self):
        env = dict(os.environ, NAO_CAMERA_VIEWER="/missing/viewer", DISPLAY=":test")
        self.assertNotEqual(subprocess.run([str(SCRIPT), "192.168.49.7"], env=env).returncode, 0)


if __name__ == "__main__":
    unittest.main()
