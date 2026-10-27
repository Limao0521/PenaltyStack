import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class BehaviorConfigCompatibilityTest(unittest.TestCase):
    def test_all_scenarios_define_keeper_sit_parameter(self):
        configs = list((ROOT / "Config/Scenarios").glob("*/behaviorParameters.cfg"))
        self.assertTrue(configs)
        missing = [str(path) for path in configs if not re.search(r"(?m)^\s*keeperSitWhenNoDive\s*=", path.read_text())]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
