"""Calibrated feet projection and pixel-only shooter association (v1).

Standard library/Python 2.7. Opponent yaw and jersey identity remain unknown.
This component consumes the real B-Human network's upper-camera detections.
"""
from __future__ import division

import math

from penalty.perception import (ShooterSelector, finite, ground, local,
                                rotate_covariance, vector)

VERSION = 'bhuman_opponent_fusion_v1'


class GroundCameraCalibration(object):
    def __init__(self, width, height, fx, fy, cx, cy, calibration_id,
                 max_range_m=4., pixel_sigma=4., rotation_sigma=.04):
        self.width, self.height = int(width), int(height)
        self.fx, self.fy, self.cx, self.cy = vector((fx, fy, cx, cy), 4)
        self.calibration_id = str(calibration_id)
        self.max_range_m, self.pixel_sigma, self.rotation_sigma = vector(
            (max_range_m, pixel_sigma, rotation_sigma), 3)
        if ((self.width, self.height) not in ((640, 480), (320, 240)) or
                self.fx <= 0 or self.fy <= 0 or not 0 <= self.cx < self.width or
                not 0 <= self.cy < self.height or not self.calibration_id or
                not .2 <= self.max_range_m <= 10 or not 1 <= self.pixel_sigma <= 20 or
                not 0 <= self.rotation_sigma <= .2):
            raise ValueError('invalid calibrated upper-camera intrinsics/uncertainty')

    def project(self, detection, camera_to_ground):
        left, top, right, bottom = vector(detection['box'], 4)
        probability = vector((detection['confidence'],), 1)[0]
        # Network boxes may extend beyond the frame. Such feet cannot give range.
        if (not .6 <= probability <= 1. or left < 0 or right >= self.width or
                right <= left or top >= bottom or bottom < 2 or bottom >= self.height-2 or
                right-left >= self.width/2):
            return None
        t = vector(camera_to_ground, 16)
        if any(abs(t[12+i]-v) > 1e-5 for i, v in enumerate((0, 0, 0, 1))):
            raise ValueError('invalid homogeneous camera transform')
        rows = [t[i:i+3] for i in (0, 4, 8)]
        for i in range(3):
            for j in range(3):
                if abs(sum(rows[i][k]*rows[j][k] for k in range(3))-float(i == j)) > 1e-4:
                    raise ValueError('camera rotation must be orthonormal')
        det = (rows[0][0]*(rows[1][1]*rows[2][2]-rows[1][2]*rows[2][1]) -
               rows[0][1]*(rows[1][0]*rows[2][2]-rows[1][2]*rows[2][0]) +
               rows[0][2]*(rows[1][0]*rows[2][1]-rows[1][1]*rows[2][0]))
        if abs(det-1) > 1e-4 or t[11] <= 0:
            raise ValueError('camera must be above ground with a proper rotation')

        def intersect(u, v):
            ray = (1., -(u-self.cx)/self.fx, -(v-self.cy)/self.fy)
            direction = [sum(rows[i][j]*ray[j] for j in range(3)) for i in range(3)]
            if direction[2] >= -1e-5:
                return None
            depth = -t[11]/direction[2]
            point = (t[3]+depth*direction[0], t[7]+depth*direction[1])
            return point if point[0] > 0 and math.hypot(*point) <= self.max_range_m else None

        middle = (left+right)/2
        center, edge_l, edge_r = intersect(middle, bottom), intersect(left, bottom), intersect(right, bottom)
        dx, dy = intersect(middle+1, bottom), intersect(middle, bottom+1)
        if any(p is None for p in (center, edge_l, edge_r, dx, dy)):
            return None
        field_width = math.hypot(edge_l[0]-edge_r[0], edge_l[1]-edge_r[1])
        if not .08 <= field_width <= .75:
            return None
        jx, jy = [dx[i]-center[i] for i in range(2)], [dy[i]-center[i] for i in range(2)]
        # Includes conservative angular-model uncertainty, beyond one-pixel rays.
        angular_variance = (self.rotation_sigma * math.hypot(*center))**2
        cov = tuple(tuple(self.pixel_sigma**2*(jx[i]*jx[j]+jy[i]*jy[j]) +
                          (angular_variance + .0001 if i == j else 0.) for j in range(2)) for i in range(2))
        return {'position': center, 'position_covariance': cov,
                'confidence': probability if detection.get('confidence_measured', True) else None,
                'confidence_measured': detection.get('confidence_measured', True),
                'box': tuple(detection['box']), 'type': 'unknown', 'fallen': False,
                'fallen_measured': False, 'field_width_m': field_width}


class OpponentFusion(object):
    def __init__(self, max_age=.25):
        self.max_age = float(max_age)
        if not finite(self.max_age) or not 0 < self.max_age <= .25:
            raise ValueError('opponent maximum age must be at most .25 s')
        self.selector = ShooterSelector(max_age=self.max_age)
        self.last_capture = None

    def reset(self):
        self.selector.reset()
        self.last_capture = None

    def merge(self, state, detection, calibration, camera_to_ground, capture_odometry,
              current_odometry, received_s, synchronized=True):
        if state['version'] != 'vision_state_v1' or state['source'] not in ('simulated_camera', 'naoqi_camera'):
            raise ValueError('camera-derived ball state required')
        state = dict(state, opponent={'position': (0., 0.), 'yaw': 0., 'position_valid': False,
                                     'yaw_valid': False, 'age_s': 40., 'track_id': None})
        if detection.get('version') != 'bhuman_robot_pixels_v1':
            raise ValueError('original robot-detector packet required')
        capture, received, decision = vector((detection['capture_s'], received_s, state['decision_s']), 3)
        if capture > received+1e-6 or received > decision+1e-6:
            raise ValueError('opponent capture <= receipt <= decision required')
        if (detection['width'], detection['height']) != (calibration.width, calibration.height):
            raise ValueError('upper-camera calibration resolution differs')
        capture_pose, current_pose = vector(capture_odometry, 3), vector(current_odometry, 3)
        trace = dict(detection, fusion=VERSION, calibration_id=calibration.calibration_id,
                     received_s=received, decision_s=decision, synchronized=bool(synchronized),
                     classification='unclassified robot; shooter candidate nearest ball',
                     field_boundary_provider=False, body_contour_provider=False)
        new = self.last_capture is None or capture > self.last_capture
        self.last_capture = max(capture, self.last_capture if self.last_capture is not None else capture)
        if not new:
            trace['reason'] = 'duplicate_or_out_of_order'
        elif not synchronized:
            trace['reason'] = 'unsynchronized_camera_and_sensors'
        elif decision-capture > self.max_age:
            trace['reason'] = 'stale'
        else:
            candidates = []
            for box in detection['boxes']:
                item = calibration.project(box, camera_to_ground)
                if item is not None:
                    item['position'] = local(ground(item['position'], capture_pose), current_pose)
                    item['position_covariance'] = rotate_covariance(item['position_covariance'], capture_pose[2]-current_pose[2])
                    item['last_seen_s'] = capture
                    candidates.append(item)
            trace['projected_candidates'] = candidates
            opponent = (self.selector.select(candidates, state['ball']['position'], decision, current_pose)
                        if state['ball']['position_valid'] else None)
            if opponent is not None:
                state = dict(state, opponent=opponent)
            trace['reason'] = 'selected' if opponent else ('ball_missing' if not state['ball']['position_valid'] else 'no_candidate_near_ball')
        # Default opponent from CameraVision is invalid, including on a miss.
        return dict(state, opponent_detector=trace)


class ConfirmedLowerFusion(OpponentFusion):
    """Lower contrast boxes have no scored confidence; require two captures."""
    def reset(self):
        super(ConfirmedLowerFusion, self).reset()
        self.previous, self.previous_capture, self.confirmations = None, None, 0

    def __init__(self):
        super(ConfirmedLowerFusion, self).__init__()
        self.reset()

    def merge(self, state, detection, calibration, camera_to_ground, capture_odometry,
              current_odometry, received_s, synchronized=True):
        state = super(ConfirmedLowerFusion, self).merge(state, detection, calibration, camera_to_ground,
                                                       capture_odometry, current_odometry, received_s,
                                                       synchronized=synchronized)
        opponent, trace = state['opponent'], state['opponent_detector']
        capture = trace['capture_s']
        if opponent['position_valid']:
            pose = vector(current_odometry, 3)
            point = ground(opponent['position'], pose)
            consistent = (self.previous is not None and 0 < capture-self.previous_capture <= .25 and
                          math.hypot(point[0]-self.previous[0], point[1]-self.previous[1]) <= .25)
            self.confirmations = self.confirmations+1 if consistent else 1
            self.previous, self.previous_capture = point, capture
            if self.confirmations < 2:
                opponent['position_valid'] = False
                trace['reason'] = 'confirming_lower_candidate'
        else:
            self.previous, self.previous_capture, self.confirmations = None, None, 0
        trace['confidence_semantics'] = 'unscored original contrast candidate, not probability 1'
        trace['confirmations'] = self.confirmations
        return state


def merge_camera_results(upper, lower):
    result = dict(lower if lower['opponent']['position_valid'] else upper)
    result['opponent_detector_upper'] = upper['opponent_detector']
    result['opponent_detector_lower'] = lower['opponent_detector']
    return result
