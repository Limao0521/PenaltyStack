import math
import unittest

import numpy as np

from penalty.keeper_fusion import FusionKeeper, FusionSettings, PoseModel, fit_track, pointing_y
from penalty.keeper_vision import VisionKeeper
from tests.test_keeper_vision import state

MODEL = PoseModel(intercept=0., slope=1., sigma=.08, ball_sigma=(.25, .15, .10, .08))


def fstate(t, x=1., y=0., vx=-1., vy=0., yaw=math.pi, yaw_valid=True, visible=True, velocity_valid=True,
           opponent=True):
    sample = state(t, x=x, y=y, vx=vx, vy=vy)
    sample['version'] = 'vision_state_v2'
    sample['ball']['visible'] = sample['ball']['position_valid'] = visible
    sample['ball']['velocity_valid'] = velocity_valid
    sample['opponent'] = {'position': (x+.2, y), 'position_valid': opponent,
                          'position_covariance': ((.005, 0.), (0., .005)), 'yaw': yaw,
                          'yaw_valid': yaw_valid and opponent, 'yaw_variance': math.radians(8)**2,
                          'age_s': 0., 'yaw_age_s': 0., 'yaw_capture_s': t, 'track_id': 1}
    return sample


def heading_to(y_target, ball=(1., 0.), goal_x=-.12):
    """Striker yaw (keeper frame) whose heading line from the ball hits y_target."""
    return math.atan2(y_target-ball[1], goal_x-ball[0])


def shot(y_cross, start=1., speed=.5, t0=1., dt=.1, frames=30, noise=0., rng=None):
    """Rolling ball from (start, 0) to the goal line at y_cross."""
    goal_x = -.12
    vy = speed*(y_cross-0.)/(start-goal_x)
    out = []
    for i in range(frames):
        t = t0+dt*i
        x, y = start-speed*dt*i, vy*dt*i
        if rng is not None:
            x, y = x+rng.normal(0, noise), y+rng.normal(0, noise)
        out.append((t, x, y, -speed, vy))
    return out


class GeometryTests(unittest.TestCase):
    def test_pointing_line_and_back_facing_striker(self):
        yaw = heading_to(.4)
        self.assertAlmostEqual(pointing_y((1., 0.), yaw, -.12), .4)
        self.assertIsNone(pointing_y((1., 0.), 0., -.12))

    def test_track_fit_recovers_crossing_time_and_rejects_static_ball(self):
        points = [(t, x, y) for t, x, y, _, _ in shot(.35, frames=6)]
        fit = fit_track(points, -.12, points[-1][0], .02)
        self.assertAlmostEqual(fit['crossing_y'], .35, places=6)
        self.assertAlmostEqual(fit['time_to_line'], (points[-1][1]+.12)/.5, places=6)
        self.assertAlmostEqual(fit['sigma'], .02, places=6)
        self.assertIsNone(fit_track([(t, 1., 0.) for t in (0., .1, .2)], -.12, .2, .02))

    def test_settings_are_bounded(self):
        with self.assertRaises(ValueError):
            FusionSettings(corner_y=.9)
        with self.assertRaises(ValueError):
            FusionSettings(fit_points=1)
        with self.assertRaises(ValueError):
            FusionKeeper('v2', use_pose=False, pose_model=MODEL)


class V4Tests(unittest.TestCase):
    def test_without_valid_yaw_v4_matches_v2_action_by_action(self):
        rng = np.random.default_rng(3)
        for episode in range(40):
            v2, v4 = VisionKeeper(), FusionKeeper('v2', True, pose_model=MODEL)
            y_cross = float(rng.uniform(-.7, .7))
            frames = [(t, 1., 0., 0., 0.) for t in np.arange(.1, 1., .1)]
            frames += shot(y_cross, t0=1., noise=.01, rng=rng, speed=float(rng.uniform(.3, 1.)))
            for t, x, y, vx, vy in frames:
                vx, vy = vx+rng.normal(0, .05), vy+rng.normal(0, .05)
                plain = state(t, x=x, y=y, vx=vx, vy=vy)
                fused = fstate(t, x=x, y=y, vx=vx, vy=vy, yaw_valid=False)
                self.assertEqual(v2(plain), v4(fused))
            self.assertEqual(v2.committed, v4.committed)

    def test_corner_prior_holds_premature_centre_and_fuses_at_dive_lead(self):
        keeper = FusionKeeper('v2', True, pose_model=MODEL)
        yaw = heading_to(.55)
        for t in np.arange(.1, 1., .1):
            keeper(fstate(t, yaw=yaw, vx=0.))
        # Early projections say centre (0.2); the prior says 0.55.
        self.assertEqual(keeper(fstate(1., x=1., y=0., vx=-.5, vy=.08*.5/1.12, yaw_valid=False))['trigger'], 0.)
        action = keeper(fstate(1.1, x=.95, y=0., vx=-.5, vy=.2*.5/1.07, yaw_valid=False))
        # v2 would commit the centre here; the fused estimate waits for a corner.
        self.assertEqual(action, {'block_side': 1, 'trigger': 0., 'amplitude': .5})
        self.assertEqual(keeper.decision['reason'], 'waiting_for_arrival')
        self.assertTrue(keeper.decision['pose_used'])
        self.assertGreater(keeper.decision['fused_y'], .3)
        # Later projections confirm the corner; commit happens at the v2 lead.
        for i, t in enumerate(np.arange(1.2, 2.5, .1)):
            x = .9-.05*i
            action = keeper(fstate(t, x=x, y=.05, vx=-.5, vy=.5*(.5-.05)/(x+.12), yaw_valid=False))
            if action['trigger']:
                break
        self.assertEqual(action, {'block_side': 1, 'trigger': 1., 'amplitude': .5})
        self.assertLessEqual(keeper.decision['time_to_line'], 1.8)

    def test_weak_corner_prior_holds_centre_until_the_dive_lead(self):
        keeper = FusionKeeper('v2', True, pose_model=PoseModel(sigma=.2, ball_sigma=(.25, .15, .10, .08)))
        for t in np.arange(.1, 1., .1):
            keeper(fstate(t, yaw=heading_to(.55), vx=0.))
        keeper(fstate(1., x=1., vx=-.5, vy=0., yaw_valid=False))
        action = keeper(fstate(1.1, x=.95, vx=-.5, vy=0., yaw_valid=False))
        self.assertLess(keeper.decision['fused_y'], .3)  # fused centre, contradicted by the prior
        self.assertEqual(action['trigger'], 0.)
        self.assertEqual(keeper.decision['reason'], 'holding_centre')
        for i, t in enumerate(np.arange(1.2, 2.5, .1)):  # held until the lead, then fused centre
            x = .9-.05*i
            action = keeper(fstate(t, x=x, y=.03, vx=-.5, vy=.5*(.05-.03)/(x+.12), yaw_valid=False))
            if action['trigger']:
                break
        self.assertEqual(action, {'block_side': 0, 'trigger': 1., 'amplitude': .5})
        self.assertLessEqual(keeper.decision['time_to_line'], 1.8)

    def test_weak_prior_does_not_hold(self):
        keeper = FusionKeeper('v2', True, pose_model=PoseModel(sigma=.5, ball_sigma=(.25, .15, .10, .08)))
        for t in np.arange(.1, 1., .1):
            keeper(fstate(t, yaw=heading_to(.55), vx=0.))
        keeper(fstate(1., x=1., vx=-.5, vy=0., yaw_valid=False))
        self.assertEqual(keeper(fstate(1.1, x=.95, vx=-.5, vy=0., yaw_valid=False))['trigger'], 1.)

    def test_centre_prior_does_not_delay_a_centre_guard(self):
        keeper = FusionKeeper('v2', True, pose_model=MODEL)
        for t in np.arange(.1, 1., .1):
            keeper(fstate(t, yaw=heading_to(.05), vx=0.))
        keeper(fstate(1., vx=-1.))
        self.assertEqual(keeper(fstate(1.1, x=.9, vx=-1.)), {'block_side': 0, 'trigger': 1., 'amplitude': .5})

    def test_stale_or_uncertain_pose_samples_are_ignored(self):
        for change in ('old', 'uncertain', 'missing_ball', 'facing_away'):
            keeper = FusionKeeper('v2', True, pose_model=MODEL)
            for t in np.arange(.1, 1., .1):
                sample = fstate(t, yaw=heading_to(.55), vx=0.)
                if change == 'old': sample['opponent']['yaw_age_s'] = .4
                if change == 'uncertain': sample['opponent']['yaw_variance'] = .5
                if change == 'missing_ball': sample['ball']['visible'] = False
                if change == 'facing_away': sample['opponent']['yaw'] = 0.
                keeper(sample)
            keeper(fstate(1., vx=-1., yaw_valid=False))
            self.assertEqual(keeper(fstate(1.1, x=.9, vx=-1., yaw_valid=False))['trigger'], 1.)
            self.assertFalse(keeper.decision['pose_used'])

    def test_commit_is_final_and_time_rewind_resets(self):
        keeper = FusionKeeper('v2', True, pose_model=MODEL)
        keeper(fstate(1., vx=-1.)); keeper(fstate(1.1, x=.9, vx=-1.))
        self.assertEqual(keeper.committed, 0)
        self.assertEqual(keeper(fstate(1.2, x=.8, y=-.6, vx=-1., vy=-.5))['block_side'], 0)
        keeper(fstate(.1, vx=0.))
        self.assertIsNone(keeper.committed)


class V5Tests(unittest.TestCase):
    def run_shot(self, keeper, y_cross, prior=None, speed=.5, noise=0., seed=0):
        rng = np.random.default_rng(seed)
        for t in np.arange(.1, 1., .1):
            keeper(fstate(t, vx=0., yaw=heading_to(prior) if prior is not None else math.pi,
                          yaw_valid=prior is not None))
        for t, x, y, vx, vy in shot(y_cross, speed=speed, noise=noise, rng=rng):
            action = keeper(fstate(t, x=x, y=y, vx=vx, vy=vy, yaw_valid=False))
            if action['trigger']:
                return action, keeper.decision
        return action, keeper.decision

    def test_centre_waits_for_lead_and_line_fit_finds_corner(self):
        keeper = FusionKeeper('v5', False, pose_model=MODEL)
        action, decision = self.run_shot(keeper, .34, noise=.005)
        self.assertEqual(action['block_side'], 1)
        self.assertLessEqual(decision['time_to_line'], 1.8)
        self.assertAlmostEqual(decision['crossing_y'], .34, delta=.03)

    def test_ball_only_v5_ignores_prior_and_full_v5_uses_it(self):
        ball_only = FusionKeeper('v5', False, pose_model=MODEL)
        _, decision = self.run_shot(ball_only, .2, prior=.6)
        self.assertFalse(decision['pose_used'])
        full = FusionKeeper('v5', True, pose_model=MODEL)
        _, decision = self.run_shot(full, .2, prior=.6)
        self.assertTrue(decision['pose_used'])
        self.assertGreater(decision['fused_y'], decision['crossing_y'])

    def test_ball_moving_away_or_lost_restarts_the_track(self):
        keeper = FusionKeeper('v5', False, pose_model=MODEL)
        for t, x, y, vx, vy in shot(.29, frames=3):  # boundary track: keeps refining
            self.assertEqual(keeper(fstate(t, x=x, y=y, vx=vx, vy=vy))['trigger'], 0.)
        keeper(fstate(1.3, x=1.2, vx=.5))
        self.assertEqual(keeper.decision['reason'], 'ball_left_track')
        self.assertEqual(keeper.track, [])
        for t, x, y, vx, vy in shot(.29, frames=3, t0=2.):
            keeper(fstate(t, x=x, y=y, vx=vx, vy=vy))
        keeper(fstate(2.8, x=.5, visible=False, velocity_valid=False))
        self.assertEqual(keeper.decision['reason'], 'track_lost')

    def test_centre_guard_waits_for_the_lead_unless_early_centre_is_enabled(self):
        keeper = FusionKeeper('v5', False, pose_model=MODEL)
        for t, x, y, vx, vy in shot(.02, frames=3):
            action = keeper(fstate(t, x=x, y=y, vx=vx, vy=vy))
        self.assertEqual(action['trigger'], 0.)
        action, decision = self.run_shot(FusionKeeper('v5', False, pose_model=MODEL), .02)
        self.assertEqual(action, {'block_side': 0, 'trigger': 1., 'amplitude': .5})
        self.assertLessEqual(decision['time_to_line'], 1.8)
        early = FusionKeeper('v5', False, FusionSettings(centre_confidence=2.), MODEL)
        for t, x, y, vx, vy in shot(.02, frames=3):
            action = early(fstate(t, x=x, y=y, vx=vx, vy=vy))
        self.assertEqual(action, {'block_side': 0, 'trigger': 1., 'amplitude': .5})
        self.assertGreater(early.decision['time_to_line'], 1.8)

    def test_published_v5_limits_are_per_side(self):
        keeper = FusionKeeper.v5(use_pose=False)
        self.assertEqual((keeper.settings.corner_y_left, keeper.settings.corner_y_right), (.35, .30))
        self.assertEqual(keeper._side(.33), 0)
        self.assertEqual(keeper._side(-.33), -1)
        self.assertEqual(FusionKeeper.v4().settings.corner_y_left, None)
        with self.assertRaises(ValueError):
            FusionSettings(corner_y_left=.05)


if __name__ == '__main__':
    unittest.main()
