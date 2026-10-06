import importlib.util
import hashlib
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo not installed')
class SimulationTests(unittest.TestCase):
    def test_original_bhuman_v6_model_is_preserved(self):
        path = Path(__file__).resolve().parents[1] / 'models' / 'bhuman' / 'NaoV6H25.rsi3'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '73f8c9a78b35fbdd18e6e407193924b2d1996812ff78cd405377cc57b0981369')
        root = ET.parse(path).getroot()
        self.assertEqual(root.find('Body').get('name'), 'Nao')
        self.assertEqual(len(list(root.find('Body').iter('Hinge'))), 24)

    def test_field_and_single_ball_impulse(self):
        from penalty.simulator import PenaltySimulator
        sim = PenaltySimulator()
        self.assertEqual(sim.model.nmesh, 100)
        self.assertAlmostEqual(sim.model.body('striker_Nao').mass[0], 1.04956, places=5)
        self.assertAlmostEqual(sum(sim.model.body_mass[i] for i in range(sim.model.nbody)
                                   if sim.model.body(i).name.startswith('striker_')),
                               5.35975, places=5)
        self.assertEqual(sim.model.joint('striker_HeadYaw').type, 3)
        self.assertEqual(sim.model.nu, 46)
        self.assertEqual(sim.model.neq, 2)
        self.assertEqual(sim.model.nbody, 52)
        self.assertAlmostEqual(sim.model.geom('orange_ball').size[0] * 2, 0.065)
        self.assertAlmostEqual(sim.model.body('ball').mass[0], 0.055)
        self.assertAlmostEqual(sim.model.geom('goal_post_left').pos[1] -
                               sim.model.geom('goal_post_right').pos[1], 1.5)
        self.assertAlmostEqual(sim.observation()['striker']['ball_range'], 0.2)
        self.assertEqual(sim.data.ncon, 1)
        sim.step(kick=True)
        self.assertEqual(sim.kick_count, 1)
        self.assertLess(sim.data.qpos[sim.ball_qpos], 0.1)
        sim.step(kick=True)
        self.assertEqual(sim.kick_count, 1)

    def test_full_control_flow(self):
        from penalty.simulator import run
        sim, states = run()
        self.assertEqual(sim.kick_count, 1)
        self.assertIn(('KICK', 'BLOCK'), states)
        self.assertEqual(states[-1], ('STOP', 'STOP'))
        self.assertGreater(sim.data.time, 28.3)
        self.assertNotAlmostEqual(sim.data.qpos[sim.ball_qpos], 0.1)

    def test_dynamic_joint_action_contract(self):
        import numpy as np
        from penalty.joint_env import JointSpaceEnv
        env = JointSpaceEnv()
        self.assertEqual(len(env.action_names), 46)
        self.assertEqual(env.model.nmocap, 0)
        obs = env.step(np.zeros(46))
        self.assertTrue(np.isfinite(obs['qpos']).all())
        self.assertGreater(obs['time_s'], 0)
        obs = env.step(np.zeros(46), repeats=990)
        self.assertAlmostEqual(obs['time_s'], 2.0)
        self.assertTrue(np.isfinite(obs['qpos']).all())
        with self.assertRaises(ValueError):
            env.step(np.zeros(26))


if __name__ == '__main__':
    unittest.main()
