"""Vision keeper v3: measured body yaw is a guarded pre-shot cue.

Ball-only v2 remains unchanged. A fresh, consistent pose prior can resolve
the 4 cm boundary of the center guard; two incoming ball captures still
confirm every commitment. This policy is simulation-only pending hardware.
"""
from collections import deque
import math

import numpy as np

from penalty.keeper_vision import VisionKeeper
from penalty.rl.anticipation import CLASSES, KickSidePredictor
from penalty.rl.orientation_contracts import from_vision_state

POLICY_VERSION = 'vision_keeper_v3'


class VisionPoseKeeper:
    def __init__(self, predictor=None, **settings):
        self.ball_keeper = VisionKeeper(**settings)
        self.predictor = predictor or KickSidePredictor.load()
        self.reset()

    def reset(self):
        self.ball_keeper.reset()
        self.pose_history = deque(maxlen=6)
        self.last_pose_capture = self.last_time = None
        self.pose_committed = False
        self.decision = {'reason': 'waiting_for_ball', 'pose_prior_used': False}

    def __call__(self, state):
        if state.get('version') != 'vision_state_v2':
            self.reset()
            self.decision['reason'] = 'invalid_version'
            return {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        try:
            from_vision_state(state)
        except (KeyError, ValueError, TypeError):
            self.reset()
            self.decision['reason'] = 'invalid_observation'
            return {'block_side': 0, 'trigger': 0., 'amplitude': .5}
        now = state['time_s']
        if self.last_time is not None and now < self.last_time:
            self.reset()
        self.last_time = now
        already_committed = self.ball_keeper.committed is not None
        action = self.ball_keeper(dict(state, version='vision_state_v1'))
        decision = dict(self.ball_keeper.decision, pose_prior_used=self.pose_committed, policy=POLICY_VERSION)
        ball, opponent = state['ball'], state['opponent']
        eligible = (ball['position_valid'] and ball['visible'] and ball['age_s'] <= .25 and
                    opponent['position_valid'] and opponent['yaw_valid'] and
                    0 <= opponent['age_s'] <= .25 and 0 <= opponent['yaw_age_s'] <= .25 and
                    0 <= opponent['yaw_variance'] <= math.radians(15)**2 and
                    max(opponent['position_covariance'][i][i] for i in (0, 1)) <= .02)
        capture = opponent.get('yaw_capture_s', -1.)
        if decision['reason'] in ('waiting_for_shot', 'unmeasured_velocity'):
            if eligible and (self.last_pose_capture is None or capture > self.last_pose_capture):
                bx, by = ball['position']
                ox, oy = opponent['position']
                yaw = opponent['yaw']
                hx, hy = math.cos(yaw), math.sin(yaw)
                pointing = by-hy*bx/hx if hx < -1e-3 else by
                features = np.array([bx, by, ox-bx, oy-by, hy, hx, pointing])
                probabilities = self.predictor.probabilities(features)
                if np.isfinite(probabilities).all():
                    self.pose_history.append((capture, probabilities))
                self.last_pose_capture = capture
        fresh = [p for stamp, p in self.pose_history if 0 <= state['decision_s']-stamp <= .5]
        decision.update(pose_measurement_eligible=bool(eligible), pose_prior_frames=len(fresh))
        if fresh:
            prior = np.mean(fresh, axis=0)
            guess = int(prior.argmax())
            decision.update(pose_predicted_side=int(CLASSES[guess]), pose_prior_probability=float(prior[guess]),
                            pose_predictor_domain='previous ideal-pose kick predictor; guarded camera use')
            crossing = decision.get('crossing_y')
            if (not already_committed and decision['reason'] == 'committed' and eligible and len(fresh) >= 2 and
                    crossing is not None and abs(abs(crossing)-self.ball_keeper.corner_y) < .04 and
                    prior[guess] >= .9 and CLASSES[guess]*crossing > 0 and
                    action['block_side'] != CLASSES[guess]):
                action = dict(action, block_side=int(CLASSES[guess]))
                self.ball_keeper.committed = action['block_side']
                self.ball_keeper.decision = dict(self.ball_keeper.decision, side=action['block_side'])
                self.pose_committed = True
                decision.update(pose_prior_used=True, side=action['block_side'])
        self.decision = decision
        return action
