"""Array adapter for the portable vision keeper. Strictly consumes rl_obs_v2."""
from penalty.keeper_vision import POLICY_VERSION, VisionKeeper
from penalty.rl.contracts import KeeperAction
from penalty.rl.vision_contracts import VisionObservation


class VisionKeeperPolicy:
    name = POLICY_VERSION
    observation_version = 'rl_obs_v2'

    def __init__(self, **settings):
        self.core = VisionKeeper(**settings)

    @property
    def decision(self):
        return self.core.decision

    def reset(self):
        self.core.reset()

    def __call__(self, observation):
        o = VisionObservation.from_array(observation)
        state = {'version': 'vision_state_v1', 'time_s': o.time_s,
                 'capture_s': o.capture_time_s, 'remaining_time': o.remaining_time,
                 'ball': {'position': (o.ball_x, o.ball_y), 'velocity': (o.ball_vx, o.ball_vy),
                          'age_s': o.ball_age, 'visible': bool(o.ball_visible),
                          'position_valid': bool(o.ball_position_valid),
                          'velocity_valid': bool(o.ball_velocity_valid),
                          'velocity_covariance': ((o.ball_velocity_x_variance, 0.), (0., 0.))}}
        return KeeperAction(**self.core(state)).to_normalized()
