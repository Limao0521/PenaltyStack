import inspect
import unittest
from penalty.control import Observation, PenaltyController


class MotionFeedbackTests(unittest.TestCase):
    def test_kick_waits_for_motor_confirmation(self):
        self.assertIn('motion_done', inspect.signature(Observation).parameters,
                      'Observation must support physical motion completion')
        controller = PenaltyController('striker')
        first = controller.update(Observation(0, start=True, ball_range=.2,
            ball_bearing=0, goal_bearing=0, motion_done=False))
        self.assertTrue(first.kick)
        waiting = controller.update(Observation(.2, motion_busy=True, motion_done=False))
        self.assertEqual(waiting.state, 'KICK')
        self.assertFalse(waiting.kick)
        finished = controller.update(Observation(.5, motion_done=True))
        self.assertEqual(finished.state, 'RECOVER')


if __name__ == '__main__':
    unittest.main()
