"""Compile and run the header-only Copa C++ tests of the B-Human overlay on the host."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1] / "bhuman/files/Tests"
COMPILER = shutil.which("g++") or shutil.which("clang++")


@unittest.skipUnless(COMPILER, "no C++ compiler")
class CopaHostCppTests(unittest.TestCase):
    def test_copa_cpp_tests_pass(self):
        sources = sorted(TESTS.glob("Copa*Test.cpp"))
        self.assertGreaterEqual(len(sources), 3)
        with tempfile.TemporaryDirectory() as tmp:
            for source in sources:
                with self.subTest(test=source.name):
                    binary = Path(tmp) / source.stem
                    build = subprocess.run([COMPILER, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source),
                                            "-o", str(binary)], capture_output=True, text=True)
                    self.assertEqual(build.returncode, 0, build.stderr)
                    self.assertEqual(subprocess.run([str(binary)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
