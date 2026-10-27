import pathlib
import subprocess
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "Make/Common/cameraStreamConfig"


class CameraStreamConfigTest(unittest.TestCase):
    def test_toggle_preserves_ports_and_changes_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "cameraStreamer.cfg"
            config.write_text("upperPort = 7777;\nlowerPort = 7778;\nenabled = true;\n")
            for value in ("off", "on"):
                subprocess.run([str(SCRIPT), str(config), value], check=True)
                self.assertEqual(config.read_text(),
                                 "upperPort = 7777;\nlowerPort = 7778;\nenabled = %s;\n" %
                                 ("true" if value == "on" else "false"))

    def test_invalid_input_does_not_change_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "cameraStreamer.cfg"
            config.write_text("enabled = true;\n")
            self.assertNotEqual(subprocess.run([str(SCRIPT), str(config), "bad"]).returncode, 0)
            self.assertEqual(config.read_text(), "enabled = true;\n")


if __name__ == "__main__":
    unittest.main()
