import importlib.util
import unittest
from types import SimpleNamespace


class RecordingWorld:
    """No controller physics: records only session scheduling boundaries."""
    dt = .012
    def __init__(self):
        self.data = SimpleNamespace(time=0.)
        self.outputs = {r: {} for r in ('striker', 'keeper')}
        self.fallen = {'striker': False, 'keeper': False}
        self.resets = []
        self.closed = False
    def reset(self, trial):
        self.resets.append(trial)
        self.data.time = 0.
    def step(self, requests):
        self.data.time += self.dt
    def stable(self):
        return True
    def observation(self):
        return {r: {'ball_range': .5, 'ball_bearing': 0., 'goal_bearing': 0.,
                    'motion_busy': False, 'motion_done': False}
                for r in ('striker', 'keeper')}
    def snapshot(self):
        return {'time_s': self.data.time, 'result': 'in_play',
                'robots': {r: {'phase': 'stand', 'fallen': False,
                               'fsr': [1.] * 8} for r in ('striker', 'keeper')}}
    def trace(self):
        return self.snapshot()
    def close(self):
        self.closed = True


class SessionTests(unittest.TestCase):
    def session(self):
        self.assertIsNotNone(importlib.util.find_spec('penalty.sim_session'),
                             'interactive trial session is missing')
        from penalty.sim_session import SimulationSession
        world = RecordingWorld()
        return SimulationSession(simulator=world, record_dir=None), world

    def start_running(self, session):
        session.command('start')
        for _ in range(250):
            session.tick()
            if session.state == 'RUNNING':
                break
        self.assertEqual(session.state, 'RUNNING')

    def test_paused_clock_and_single_step_then_complete_reset(self):
        session, world = self.session()
        self.assertEqual(session.state, 'PREPARED')
        session.tick()
        self.assertEqual(world.data.time, 0)
        self.start_running(session)
        session.command('pause')
        previous = world.data.time
        for _ in range(200):
            session.tick()
        self.assertEqual(world.data.time, previous)
        session.command('step')
        session.tick()
        self.assertAlmostEqual(world.data.time-previous, .012)
        self.assertEqual(session.state, 'PAUSED')
        session.command('reset')
        self.assertEqual(world.data.time, 0)
        self.assertEqual(session.state, 'PREPARED')
        self.assertEqual(session.snapshot()['penalty_time_s'], 0)
        self.assertEqual(len(set(world.resets)), len(world.resets))

    def test_repeat_starts_new_trial_and_abort_cannot_be_resumed(self):
        session, world = self.session()
        self.start_running(session)
        first = session.trial
        session.command('repeat')
        self.assertNotEqual(session.trial, first)
        self.assertEqual(session.state, 'SETTLING')
        self.assertAlmostEqual(world.data.time, 0)
        for _ in range(250):
            session.tick()
        session.command('abort')
        for _ in range(180):
            session.tick()
        self.assertEqual(session.state, 'ABORTED')
        session.command('pause')
        self.assertEqual(session.state, 'ABORTED')
        self.assertTrue(all(c.abort_latched for c in session.controllers.values()))
        session.close()
        session.tick()
        self.assertTrue(world.closed)
        self.assertEqual(session.state, 'CLOSED')

    def test_speed_changes_wall_pacing_only_and_rejects_nonfinite(self):
        session, world = self.session()
        self.start_running(session)
        session.command('speed', 2)
        initial = world.data.time
        session.tick()
        self.assertAlmostEqual(world.data.time-initial, .012)
        self.assertEqual(session.snapshot()['speed'], 2)
        with self.assertRaises(ValueError):
            session.command('speed', float('nan'))

    def test_kick_length_is_bounded_and_reaches_request(self):
        session,_=self.session()
        session.mode='manual'
        session.command('manual',{'motion':'kick','kick_length_mm':2000.})
        self.assertEqual(session._requests()['striker']['kick_length_mm'],2000.)
        for length in (1499.,3501.,float('nan')):
            with self.assertRaises(ValueError):
                session.command('manual',{'motion':'kick','kick_length_mm':length})

    def test_guard_is_latched_for_one_shot_despite_stopped_ball_and_new_bearing(self):
        session, world = self.session()
        self.start_running(session)
        ball = {'ball_range':.5,'ball_bearing':0.,'goal_bearing':0.,'ball_speed':.8}
        world.observation = lambda:{r:dict(ball) for r in ('striker','keeper')}
        session._next_control = 0.
        first = session._requests()['keeper']
        self.assertEqual((first['motion'],first['block_side']),('block','center'))
        ball.update(ball_speed=0.,ball_bearing=-.5)
        world.data.time += .12
        later = session._requests()['keeper']
        self.assertEqual(session.commands['keeper'].block,'right')
        self.assertEqual((later['motion'],later['block_side']),('block','center'))
        session.command('repeat')
        self.assertIsNone(session._keeper_block)

    def test_abort_transition_does_not_consume_active_penalty_clock(self):
        session, world = self.session()
        self.start_running(session)
        for _ in range(20):
            session.tick()
        elapsed = session.penalty_time
        session.command('abort')
        for _ in range(100):
            session.tick()
        self.assertGreater(world.data.time,2.)
        self.assertEqual(session.penalty_time,elapsed)

    def test_lateral_jump_releases_request_only_after_original_phase_starts(self):
        session, world = self.session()
        session.mode,session.role = 'manual','keeper'
        session.command('manual',{'motion':'block','block_side':'right'})
        self.start_running(session)
        self.assertEqual(session._requests()['keeper']['motion'],'block')
        world.outputs['keeper'] = {'phase':'keyframeMotion',
            'motion_info':{'keyframe':'keeperJumpLeft','mirror':True}}
        session.tick()
        self.assertEqual(session._requests()['keeper']['motion'],'getup')
        # Completion must not reissue the same held jump. A new explicit
        # button press is required to execute another jump.
        world.outputs['keeper'] = {'phase':'stand','motion_info':{'keyframe':'keeperJumpLeft'}}
        session.tick()
        self.assertEqual(session._requests()['keeper']['motion'],'getup')
        session.command('manual',{'motion':'block','block_side':'left'})
        self.assertEqual(session._requests()['keeper']['motion'],'block')
        session.command('reset')
        self.assertEqual(session._requests()['keeper']['motion'],'block')

    def test_automatic_lateral_jump_recovers_without_repeating_for_same_shot(self):
        session,world = self.session()
        self.start_running(session)
        ball = {'ball_range':.5,'ball_bearing':.5,'goal_bearing':0.,'ball_speed':.8}
        world.observation = lambda:{r:dict(ball) for r in ('striker','keeper')}
        first = session._requests()['keeper']
        self.assertEqual((first['motion'],first['block_side']),('block','left'))
        world.outputs['keeper'] = {'phase':'keyframeMotion',
            'motion_info':{'keyframe':'keeperJumpLeft','mirror':False}}
        session.tick()
        self.assertEqual(session._requests()['keeper']['motion'],'getup')
        ball.update(ball_speed=0.,ball_bearing=-.5)
        world.data.time += .12
        self.assertEqual(session._requests()['keeper']['motion'],'getup')


if __name__ == '__main__':
    unittest.main()
