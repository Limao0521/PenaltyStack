"""Expanded native requests retain the original session lifecycle and tooling."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from penalty.keeper_defenses import BLOCK_REQUEST_NAMES, DEFENSE_MOTIONS
from penalty.sim_session import SimulationSession
from tests.test_sim_session import RecordingWorld
from tools.build_motion_runtime import cmake_cache, visual_studio


class KeeperDefenseRuntimeTests(unittest.TestCase):
    def test_manual_catalog_accepts_only_known_requests(self):
        for defense in BLOCK_REQUEST_NAMES:
            with self.subTest(defense=defense):
                request = SimulationSession._validate_manual({'motion': 'block', 'block_side': defense})
                self.assertEqual(request['block_side'], defense)
        with self.assertRaises(ValueError):
            SimulationSession._validate_manual({'motion': 'block', 'block_side': 'unknown_guard'})

    def test_only_lateral_jumps_release_to_getup(self):
        for defense in BLOCK_REQUEST_NAMES:
            with self.subTest(defense=defense):
                world = RecordingWorld()
                session = SimulationSession(simulator=world, record_dir=None)
                session.mode, session.role = 'manual', 'keeper'
                session.command('manual', {'motion': 'block', 'block_side': defense})
                session.command('start')
                for _ in range(250):
                    session.tick()
                    if session.state == 'RUNNING':
                        break
                self.assertEqual(session.state, 'RUNNING')
                self.assertEqual(session._requests()['keeper']['motion'], 'block')
                # A completed/accepted guard does not release a squat or its
                # preparation. Only original lateral jumps have auto-getup.
                world.outputs['keeper'] = {'phase': 'keyframeMotion',
                    'motion_info': {'keyframe': DEFENSE_MOTIONS[defense][1],
                                    'mirror': DEFENSE_MOTIONS[defense][2]}}
                session.tick()
                request = session._requests()['keeper']
                self.assertEqual(request['block_side'], defense)
                self.assertEqual(request['motion'], 'getup' if defense in ('left', 'right') else 'block')
                session.close()
                session.tick()

    def test_builder_uses_latest_supported_installed_visual_studio(self):
        installs = [
            {'installationVersion': '17.9.1', 'installationPath': 'C:/VS/2022'},
            {'installationVersion': '18.0.1', 'installationPath': 'C:/VS/2026'},
        ]
        self.assertEqual(visual_studio(installs, {}), ('C:/VS/2026', 'Visual Studio 18 2026'))
        self.assertEqual(visual_studio(installs, {'CMAKE_GENERATOR': 'Visual Studio 17 2022'}),
                         ('C:/VS/2022', 'Visual Studio 17 2022'))

    def test_builder_preserves_cache_instance_and_rejects_missing_instance(self):
        installs = [
            {'installationVersion': '18.0.1', 'installationPath': 'C:/VS/2026'},
            {'installationVersion': '18.0.2', 'installationPath': 'C:/VS/2026Other'},
        ]
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / 'CMakeCache.txt'
            cache_path.write_text('CMAKE_GENERATOR:INTERNAL=Visual Studio 18 2026\n'
                                  'CMAKE_GENERATOR_INSTANCE:INTERNAL=C:/VS/2026Other\n')
            cache = cmake_cache(cache_path)
            self.assertEqual(visual_studio(installs, cache), ('C:/VS/2026Other', 'Visual Studio 18 2026'))
            cache['CMAKE_GENERATOR_INSTANCE'] = 'C:/VS/Missing'
            with self.assertRaises(RuntimeError):
                visual_studio(installs, cache)


if __name__ == '__main__':
    unittest.main()
