"""The NAO's C++ tactical layer must reproduce the published Python policies.

Builds native/copa_policy_tests (host compiler) and replays tests/robot/copa_golden.json,
written by tools/export_robot_policies.py. Also checks that the exported robot .cfg
files hold exactly the golden weights and that the deploy registry is consistent.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

import numpy

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / 'native/copa_policy_tests'
BUILD = HARNESS / 'build'
GOLDEN = ROOT / 'tests/robot/copa_golden.json'
POLICIES = ROOT / 'robot/Config/NeuralNets/CopaPenalty'


def find_cmake():
    cmake = shutil.which('cmake')
    if cmake or os.name != 'nt':
        return cmake, []
    vswhere = Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
    if not vswhere.is_file():
        return None, []
    found = json.loads(subprocess.check_output(
        [str(vswhere), '-latest', '-products', '*', '-requires',
         'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-format', 'json'], text=True))
    if not found:
        return None, []
    install = found[0]
    generator = {17: 'Visual Studio 17 2022', 18: 'Visual Studio 18 2026'}.get(
        int(install['installationVersion'].split('.')[0]))
    cmake = Path(install['installationPath']) / 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
    return (str(cmake), ['-G', generator, '-A', 'x64']) if cmake.is_file() and generator else (None, [])


def build_harness():
    cmake, generator = find_cmake()
    if cmake is None:
        raise unittest.SkipTest('CMake/compiler not available')
    if not (ROOT / 'native/bhuman_motion/dependencies/json.hpp').is_file():
        raise unittest.SkipTest('run tools/build_motion_runtime.py first (json.hpp)')
    subprocess.run([cmake, '-S', str(HARNESS), '-B', str(BUILD), *generator, '-DCMAKE_BUILD_TYPE=Release'],
                   check=True, capture_output=True, text=True)
    subprocess.run([cmake, '--build', str(BUILD), '--config', 'Release'], check=True, capture_output=True, text=True)
    for candidate in (BUILD / 'Release/copa_policy_tests.exe', BUILD / 'copa_policy_tests'):
        if candidate.is_file():
            return candidate
    raise AssertionError('harness executable not found')


def cfg_numbers(text, key):
    """Numbers of `key = [...]` (top level) in a B-Human cfg written by the exporter."""
    match = re.search(r'^%s = \[(.*?)\];' % re.escape(key), text, re.S | re.M)
    return [float(v) for v in re.findall(r'[-+0-9.eE]+', match.group(1))]


class RobotPolicyParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(GOLDEN.read_text(encoding='utf-8'))

    def test_cpp_header_matches_python_policies(self):
        try:
            executable = build_harness()
        except subprocess.CalledProcessError as error:
            self.fail('harness build failed:\n%s\n%s' % (error.stdout, error.stderr))
        result = subprocess.run([str(executable), str(GOLDEN)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('failures=0', result.stdout)

    def test_robot_cfg_holds_golden_weights(self):
        for name, entry in self.golden['strikers'].items():
            text = (POLICIES / (name + '.cfg')).read_text(encoding='utf-8')
            policy = entry['policy']
            self.assertEqual(cfg_numbers(text, 'obsMean'), policy['obsMean'], name)
            self.assertEqual(cfg_numbers(text, 'obsVar'), policy['obsVar'], name)
            # The robot reads weights as float32; nine digits round-trip every float32 exactly.
            weights = [float(v) for v in re.findall(r'weights = \[(.*?)\];', text, re.S)[0]
                       .replace('\n', ' ').split(',')]
            expected = [v for row in policy['hidden'][0]['weights'] for v in row]
            self.assertEqual(numpy.asarray(weights, dtype=numpy.float32).tolist(),
                             numpy.asarray(expected, dtype=numpy.float32).tolist(), name)

    def test_registry_matches_exported_files(self):
        registry = json.loads((POLICIES / 'policies.json').read_text(encoding='utf-8'))
        self.assertIn(registry['defaults']['striker'], registry['striker'])
        self.assertIn(registry['defaults']['keeper'], registry['keeper'])
        for role in ('striker', 'keeper'):
            for name, entry in registry[role].items():
                if entry['file']:
                    data = (POLICIES / entry['file']).read_bytes()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'], name)


if __name__ == '__main__':
    unittest.main()
