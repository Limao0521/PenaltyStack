import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo not installed')
class SimulationTests(unittest.TestCase):
    def test_field_and_single_ball_impulse(self):
        from penalty.simulator import PenaltySimulator
        sim = PenaltySimulator()
        self.assertAlmostEqual(sim.model.geom('orange_ball').size[0] * 2, 0.065)
        self.assertAlmostEqual(sim.model.body('ball').mass[0], 0.055)
        self.assertAlmostEqual(sim.model.geom('goal_post_left').pos[1] -
                               sim.model.geom('goal_post_right').pos[1], 1.5)
        self.assertAlmostEqual(sim.observation()['striker']['ball_range'], 0.2)
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


if __name__ == '__main__':
    unittest.main()
