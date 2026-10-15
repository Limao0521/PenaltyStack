"""Deploy tooling and robot scenario of the Copa NAO CCM MX 2026 penalty routine."""
import os
from pathlib import Path
import re
import shutil
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
sys.path.insert(0, str(ROOT / 'tools'))

import check_module_graph  # noqa: E402
import robot_build  # noqa: E402
import robot_deploy  # noqa: E402

SCENARIO = ROOT / 'robot/Config/Scenarios/CopaNAOPenalty'


def cfg_value(path, key):
    match = re.search(r'^%s\s*=\s*([^;]+);' % key, path.read_text(encoding='utf-8'), re.M)
    return match.group(1).strip().strip('"') if match else None


class RobotRegistryTest(unittest.TestCase):
    def test_team_robots_have_wired_addresses(self):
        robots = robot_deploy.robots()
        self.assertGreaterEqual(len(robots), 3, 'rules 3.3.3: three robots must shoot')
        for robot in robots:
            self.assertRegex(robot.lan, r'^\d+\.\d+\.\d+\.\d+$', robot.name)

    def test_policy_registry_defaults_are_the_recommended_ones(self):
        registry = robot_deploy.policies()
        self.assertEqual(registry['defaults'], {'striker': 'striker_v2', 'keeper': 'vision_v2_balanced'})
        self.assertEqual(registry['striker']['striker_v2']['status'], 'recomendado')
        self.assertEqual(registry['keeper']['vision_v2_balanced']['status'], 'recomendado')
        for name, entry in registry['striker'].items():
            if entry['file']:
                self.assertTrue((ROOT / 'robot/Config/NeuralNets/CopaPenalty' / entry['file']).is_file(), name)


class DeployCommandTest(unittest.TestCase):
    def test_command_selects_role_policies_and_switches_wifi_off(self):
        robot = robot_deploy.Robot('loki', '192.168.49.3', '')
        options = robot_deploy.DeployOptions(role='keeper', striker_policy='striker_preserving_v4',
                                             keeper_policy='vision_v2_original', start_mode='buttonOrWhistle',
                                             player=4, team=49, jumps=False)
        command = robot_deploy.deploy_command(robot, options)
        for part in ('./deployCopaPenalty Release 192.168.49.3', '--role keeper', '--striker-policy striker_preserving_v4',
                     '--keeper-policy vision_v2_original', '--start-mode buttonOrWhistle', '--jumps off',
                     '-w NONE', '-p 4', '-t 49', '-b'):
            self.assertIn(part, command)

    def test_invalid_options_are_rejected_before_any_remote_step(self):
        robot = robot_deploy.Robot('loki', '192.168.49.3', '')
        for options in (robot_deploy.DeployOptions(role='goalie'),
                        robot_deploy.DeployOptions(striker_policy='striker_v9'),
                        robot_deploy.DeployOptions(keeper_policy='keeper_v4'),
                        robot_deploy.DeployOptions(start_mode='voice'),
                        robot_deploy.DeployOptions(player=0)):
            with self.assertRaises(ValueError):
                robot_deploy.deploy_command(robot, options)

    def test_wsl_paths(self):
        if os.name == 'nt':
            self.assertEqual(robot_build.to_wsl_path(Path('C:/Users/x/robot')), '/mnt/c/Users/x/robot')


class ScenarioTest(unittest.TestCase):
    def test_module_graph_is_complete(self):
        errors, used = check_module_graph.check('CopaNAOPenalty', 'CopaNAO_CCM2026')
        self.assertEqual(errors, [])
        for module in ('CopaPenaltyController', 'CopaPenaltyTactics', 'CopaOrangeBallPerceptor', 'SkillBehaviorControl'):
            self.assertIn(module, used)
        for module in ('GameStateProvider', 'YoloBallDetector'):
            self.assertNotIn(module, used)

    def test_checker_finds_a_missing_provider(self):
        original = check_module_graph.parse_threads

        def without_ball_spots(path):
            defaults, threads = original(path)
            return [d for d in defaults if d != 'BallSpots'], threads
        with mock.patch.object(check_module_graph, 'parse_threads', without_ball_spots):
            errors, _ = check_module_graph.check('CopaNAOPenalty', 'CopaNAO_CCM2026')
        self.assertTrue(any('BallSpots' in error for error in errors), errors)

    def test_original_scenarios_stay_valid(self):
        for scenario, location in (('Default', '4v4_Full'), ('4v4_Complete', '4v4_Full')):
            self.assertEqual(check_module_graph.check(scenario, location)[0], [], scenario)

    def test_competition_safe_defaults(self):
        controller = SCENARIO / 'copaPenaltyController.cfg'
        self.assertEqual(cfg_value(controller, 'startMode'), 'buttonOnly')
        self.assertLessEqual(int(cfg_value(controller, 'activeDurationMs') or 0), 30000, 'rule 3.3.5: 30 s')
        tactics = SCENARIO / 'copaPenaltyTactics.cfg'
        self.assertEqual(cfg_value(tactics, 'penaltyMarkToGoalLine'), '1300')
        self.assertEqual(cfg_value(tactics, 'strikerBehindMark'), '200')
        self.assertEqual(cfg_value(tactics, 'goalWidth'), '1500')
        ball = ROOT / 'robot/Config/Locations/CopaNAO_CCM2026/ballSpecification.cfg'
        self.assertEqual(cfg_value(ball, 'radius'), '32.5')
        self.assertEqual(cfg_value(SCENARIO / 'behaviorParameters.cfg', 'keeperJumpingOn'), 'true')
        self.assertEqual(cfg_value(SCENARIO / 'strategyBehaviorControl.cfg', 'enableEmbeddedPPO'), 'false')

    def test_only_documented_originals_were_modified(self):
        changed, missing, _ = robot_build.modified_originals()
        self.assertEqual(missing, [])
        allowed = {'Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.h'}
        for name in changed:
            self.assertTrue(name in allowed or re.match(r'^Config/Scenarios/[^/]+/threads\.cfg$', name), name)


@unittest.skipUnless(os.name == 'nt' and shutil.which('wsl.exe'), 'WSL deploy test')
class DeployScriptDryRunTest(unittest.TestCase):
    def test_dry_run_stages_the_robot_configuration(self):
        probe = robot_build.run_bash('test -x %s/Build/Linux/Nao/Release/bhuman' % robot_build.WORK, check=False, capture=True)
        if probe.returncode != 0:
            self.skipTest('build the robot first: python tools/robot_build.py')
        self.assertEqual(robot_deploy.sync(lambda _line: None), 0)
        lines = []
        robot = robot_deploy.Robot('falcanao', '192.168.49.4', '')
        options = robot_deploy.DeployOptions(role='keeper', striker_policy='scripted', keeper_policy='center_guard',
                                             player=3, team=49, dry_run=True)
        self.assertEqual(robot_deploy.deploy(robot, options, lines.append), 0, '\n'.join(lines))
        text = '\n'.join(lines)
        for expected in ('scenario = CopaNAOPenalty;', 'location = CopaNAO_CCM2026;', 'playerNumber = 3;',
                         'teamNumber = 49;', 'role = keeper;', 'strikerPolicy = "scripted";',
                         'keeperPolicy = "center_guard";', 'setprofile NONE', 'DEPLOY_OK falcanao'):
            self.assertIn(expected, text)


if __name__ == '__main__':
    unittest.main()
