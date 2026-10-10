import json
import pathlib
import tarfile
import tempfile
import unittest

from deploy.package import package


class PackageTests(unittest.TestCase):
    def test_isolated_reproducible_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            profile = root / 'profile.json'
            profile.write_text(json.dumps({'role': 'striker', 'robot_id': 'TEST-NAO-01',
                                           'motion_enabled': False, 'kick_enabled': False}))
            a, b = root / 'a.tar.gz', root / 'b.tar.gz'
            self.assertEqual(package(profile, a), package(profile, b))
            with tarfile.open(a, 'r:gz') as archive:
                names = archive.getnames()
                self.assertIn('penalty/control.py', names)
                self.assertIn('penalty/perception.py', names)
                self.assertIn('penalty/keeper_vision.py', names)
                self.assertIn('penalty/nao_vision.py', names)
                self.assertIn('profile.json', names)
                self.assertNotIn('sim/scene.xml', names)
                self.assertFalse(any(name.startswith('heritage/') for name in names))

    def test_vision_package_rejects_enabled_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            profile = root / 'profile.json'
            profile.write_text(json.dumps({'role': 'keeper', 'robot_id': 'TEST-NAO-01',
                                          'keeper_policy': 'vision', 'block_enabled': True}))
            with self.assertRaises(ValueError):
                package(profile, root / 'release.tar.gz')

    def test_rejects_unprepared_robot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            profile = root / 'profile.json'
            profile.write_text(json.dumps({'role': 'striker', 'robot_id': 'TO_FILL',
                                           'kick_enabled': False}))
            with self.assertRaises(ValueError):
                package(profile, root / 'release.tar.gz')


if __name__ == '__main__':
    unittest.main()
