import unittest

from penalty.control import Observation, PenaltyController


class ControllerTests(unittest.TestCase):
    def test_wait_abort_and_deadline(self):
        c = PenaltyController('striker')
        self.assertEqual(c.update(Observation(0)).state, 'WAIT_FOR_START')
        self.assertEqual(c.update(Observation(1, start=True)).state, 'SEARCH_BALL')
        self.assertEqual(c.update(Observation(29)).state, 'STOP')
        self.assertFalse(c.update(Observation(29.1, start=True)).kick)
        k = PenaltyController('keeper')
        self.assertEqual(k.update(Observation(0, abort=True)).state, 'STOP')

    def test_striker_one_kick_then_stop(self):
        c = PenaltyController('striker')
        c.update(Observation(0, start=True))
        self.assertEqual(c.update(Observation(0.1, ball_range=0.6, ball_bearing=0)).state, 'APPROACH_BALL')
        action = c.update(Observation(0.2, ball_range=0.18, ball_bearing=0, goal_bearing=0))
        self.assertTrue(action.kick)
        self.assertEqual(c.update(Observation(0.3)).state, 'RECOVER')
        self.assertEqual(c.update(Observation(2.0)).state, 'STOP')
        self.assertFalse(c.update(Observation(3, ball_range=0.18, ball_bearing=0, goal_bearing=0)).kick)

    def test_stale_ball_and_goal_guard(self):
        c = PenaltyController('striker')
        c.update(Observation(0, start=True))
        self.assertEqual(c.update(Observation(1, ball_range=0.18, ball_bearing=0, ball_age=1)).state, 'SEARCH_BALL')
        self.assertEqual(c.update(Observation(2, ball_range=0.18, ball_bearing=0)).state, 'ESTIMATE_GOAL')
        self.assertFalse(c.kicked)

    def test_keeper_tracks_without_walking(self):
        c = PenaltyController('keeper')
        c.update(Observation(0, start=True))
        a = c.update(Observation(1, ball_range=0.8, ball_bearing=0.3))
        self.assertEqual(a.block, 'left')
        self.assertEqual(a.lateral, 0)
        self.assertEqual(c.update(Observation(1.1, abort=True)).state, 'STOP')

    def test_approach_budget_stops_false_far_ball(self):
        c = PenaltyController('striker')
        c.update(Observation(0, start=True))
        for tick in range(1, 100):
            action = c.update(Observation(tick * 0.1, ball_range=5, ball_bearing=0))
            if action.state == 'STOP':
                break
        self.assertEqual(c.state, 'STOP')
        self.assertLess(c.commanded_forward, 0.65)


if __name__ == '__main__':
    unittest.main()
