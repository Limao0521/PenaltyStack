"""Local camera/model observations. Standard library only; Python 2.7 compatible.

Ground frame: x forward, y left, metres. Camera frame: x optical-forward,
y left, z up (NAOqi). Capture, receipt and decision use one monotonic clock.
No scene ball/opponent positions are read by this module.
"""
from __future__ import division

import math

VISION_VERSION = 'vision_state_v1'
SOURCES = ('naoqi_camera', 'simulated_camera', 'bhuman_models', 'scene_privileged')


def finite(value):
    return isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value)


def vector(values, size):
    result = tuple(float(v) for v in values)
    if len(result) != size or not all(finite(v) for v in result):
        raise ValueError('expected %d finite values' % size)
    return result


def local(point, pose):
    dx, dy = point[0] - pose[0], point[1] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return c * dx + s * dy, -s * dx + c * dy


def ground(point, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pose[0] + c * point[0] - s * point[1], pose[1] + s * point[0] + c * point[1]


def rotate_covariance(cov, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    a, b, d = cov[0][0], cov[0][1], cov[1][1]
    return ((c*c*a - 2*c*s*b + s*s*d, c*s*(a-d) + (c*c-s*s)*b),
            (c*s*(a-d) + (c*c-s*s)*b, s*s*a + 2*c*s*b + c*c*d))


def covariance(values, scale=1.):
    a, b = vector(values[0], 2), vector(values[1], 2)
    if a[0] < 0 or b[1] < 0 or abs(a[1]-b[0]) > 1e-6 or a[0]*b[1] < a[1]*b[0]-1e-12:
        raise ValueError('covariance must be symmetric positive semidefinite')
    return ((a[0]*scale, a[1]*scale), (b[0]*scale, b[1]*scale))


class CameraCalibration(object):
    def __init__(self, width, height, fx, fy, cx, cy, ball_radius_m,
                 calibration_id, max_range_m=4., pixel_sigma=1.5):
        self.width, self.height = int(width), int(height)
        self.fx, self.fy, self.cx, self.cy = vector((fx, fy, cx, cy), 4)
        self.ball_radius_m = float(ball_radius_m)
        self.calibration_id = str(calibration_id)
        self.max_range_m, self.pixel_sigma = float(max_range_m), float(pixel_sigma)
        if (self.width < 2 or self.height < 2 or self.fx <= 0 or self.fy <= 0 or
                not 0 <= self.cx <= self.width or not 0 <= self.cy <= self.height or
                not finite(self.ball_radius_m) or not .01 <= self.ball_radius_m <= .1 or
                not self.calibration_id or not .2 <= self.max_range_m <= 10 or
                not .1 <= self.pixel_sigma <= 10):
            raise ValueError('invalid camera calibration')

    def project_ball(self, candidate, camera_to_ground):
        """Intersect centroid ray with z=ball radius; reject clipped/implausible blobs."""
        t = vector(camera_to_ground, 16)
        # Invalid sensor/calibration transforms must not fabricate a ball.
        if any(abs(t[12+i] - v) > 1e-5 for i, v in enumerate((0., 0., 0., 1.))):
            raise ValueError('invalid homogeneous camera transform')
        rows = [t[i:i+3] for i in (0, 4, 8)]
        for i in range(3):
            for j in range(3):
                if abs(sum(rows[i][k]*rows[j][k] for k in range(3)) - float(i == j)) > 1e-4:
                    raise ValueError('camera rotation must be orthonormal')

        def intersect(u, v):
            ray = (1., -(u-self.cx)/self.fx, -(v-self.cy)/self.fy)
            direction = [sum(rows[i][j]*ray[j] for j in range(3)) for i in range(3)]
            if direction[2] >= -1e-5 or t[11] <= self.ball_radius_m:
                return None
            distance = (self.ball_radius_m-t[11])/direction[2]
            xy = (t[3]+distance*direction[0], t[7]+distance*direction[1])
            if distance <= 0 or xy[0] <= 0 or math.hypot(*xy) > self.max_range_m:
                return None
            return xy, distance

        u, v = candidate['center']
        hit = intersect(u, v)
        if hit is None:
            return None
        xy, depth = hit
        expected_diameter = 2*self.ball_radius_m * (self.fx+self.fy)/2 / depth
        # Truncated balls/feet and large orange patches are not range measurements.
        if not .45 <= candidate['diameter']/expected_diameter <= 1.8:
            return None
        xp, yp = intersect(u+1, v), intersect(u, v+1)
        if xp is None or yp is None:
            return None
        dx, dy = [xp[0][i]-xy[i] for i in range(2)], [yp[0][i]-xy[i] for i in range(2)]
        variance = self.pixel_sigma**2
        cov = tuple(tuple(variance*(dx[i]*dx[j]+dy[i]*dy[j]) + (1e-6 if i == j else 0.)
                          for j in range(2)) for i in range(2))
        return {'position': xy, 'covariance': cov, 'candidate': candidate}


def orange_candidate(rgb, width, height, stride=2, min_pixels=3):
    """Pixel-only detector; retains centroid/bounds needed for calibrated projection."""
    if width < 2 or height < 2 or stride < 1 or len(rgb) != width*height*3:
        raise ValueError('RGB dimensions/buffer mismatch')
    cells = set()
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            index = (y*width+x)*3
            r, g, b = rgb[index:index+3]
            if not isinstance(r, int):
                r, g, b = ord(r), ord(g), ord(b)
            if r >= 115 and 45 <= g <= 200 and b <= 120 and r > g*1.18 and g > b*1.15:
                cells.add((x//stride, y//stride))
    best = None
    while cells:
        seed = cells.pop()
        queue, group = [seed], [seed]
        while queue:
            x, y = queue.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (x+dx, y+dy)
                    if neighbor in cells:
                        cells.remove(neighbor)
                        queue.append(neighbor)
                        group.append(neighbor)
        if best is None or len(group) > len(best):
            best = group
    if not best or len(best) < min_pixels:
        return None
    xs, ys = [p[0]*stride for p in best], [p[1]*stride for p in best]
    x0, x1, y0, y1 = min(xs), max(xs)+stride, min(ys), max(ys)+stride
    w, h = x1-x0, y1-y0
    if (x0 == 0 or y0 == 0 or x1 >= width or y1 >= height or
            max(w/h, h/w) > 1.8 or len(best)*stride**2/(w*h) < .45):
        return None
    return {'center': (sum(xs)/len(xs)+(stride-1)/2, sum(ys)/len(ys)+(stride-1)/2),
            'bounds': (x0, y0, x1, y1), 'diameter': (w+h)/2., 'pixels': len(best)}


class BallTracker(object):
    """Estimate velocity in odometry frame, then rotate to current robot frame.

    Two independent captures are needed. A gap/outlier restarts velocity;
    duplicates/out-of-order captures cannot confirm a shot. No root motion.
    """
    def __init__(self, max_age=.25, max_speed=8., max_gap=.25):
        self.max_age, self.max_speed, self.max_gap = max_age, max_speed, max_gap
        self.reset()

    def reset(self):
        self.last_capture = None
        self.last_seen = None
        self.point = self.point_cov = self.last_pose = None
        self.velocity, self.velocity_cov = (0., 0.), ((0., 0.), (0., 0.))
        self.velocity_valid = False
        self.reason = 'unseen'
        self.received_s = None

    def update(self, capture_s, received_s, measurement, odometry):
        capture_s, received_s = vector((capture_s, received_s), 2)
        pose = vector(odometry, 3)
        if capture_s > received_s + 1e-6:
            raise ValueError('capture is in the future; clocks must be aligned')
        if self.last_capture is not None and capture_s <= self.last_capture:
            self.reason = 'duplicate' if capture_s == self.last_capture else 'out_of_order'
            return False
        self.last_capture, self.received_s = capture_s, received_s
        if received_s-capture_s > self.max_age:
            self.reason = 'delayed_frame'
            return False
        if measurement is None:
            self.reason = 'not_seen'
            return False
        point = ground(vector(measurement['position'], 2), pose)
        cov = rotate_covariance(covariance(measurement['covariance']), pose[2])
        self.velocity_valid = False
        if self.last_seen is not None:
            dt = capture_s-self.last_seen
            if .025 <= dt <= self.max_gap:
                velocity = tuple((point[i]-self.point[i])/dt for i in range(2))
                if math.hypot(*velocity) <= self.max_speed:
                    self.velocity = velocity
                    self.velocity_cov = tuple(tuple((cov[i][j]+self.point_cov[i][j])/dt**2
                                                    for j in range(2)) for i in range(2))
                    self.velocity_valid = True
                else:
                    self.reason = 'position_outlier'
                    # Do not let one rejected candidate contaminate the next derivative.
                    self.last_seen = self.point = self.point_cov = None
                    return False
        self.point, self.point_cov, self.last_pose, self.last_seen = point, cov, pose, capture_s
        self.reason = 'tracked' if self.velocity_valid else 'position_only'
        return True

    def estimate(self, now_s, odometry):
        pose = vector(odometry, 3)
        age = max(0., now_s-self.last_seen) if self.last_seen is not None else 40.
        valid = self.point is not None and 0 <= now_s-self.last_seen <= self.max_age
        point = self.point or (0., 0.)
        # Position is advanced only within the measured track's freshness window.
        if valid and self.velocity_valid:
            point = tuple(point[i]+self.velocity[i]*age for i in range(2))
        velocity = local(self.velocity, (0., 0., pose[2]))
        return {'position': local(point, pose), 'velocity': velocity,
                'position_valid': bool(valid), 'velocity_valid': bool(valid and self.velocity_valid),
                'visible': bool(valid and self.reason in ('tracked', 'position_only')),
                'age_s': age, 'last_seen_s': self.last_seen,
                'position_covariance': rotate_covariance(self.point_cov or ((0., 0.), (0., 0.)), -pose[2]),
                'velocity_covariance': rotate_covariance(self.velocity_cov, -pose[2]),
                'reason': self.reason}


def vision_state(source, capture_s, received_s, decision_s, time_s, ball,
                 opponent=None, calibration_id=None, sensors=None):
    if source not in SOURCES:
        raise ValueError('unknown perception source')
    capture_s, received_s, decision_s, time_s = vector((capture_s, received_s, decision_s, time_s), 4)
    if capture_s > received_s+1e-6 or received_s > decision_s+1e-6:
        raise ValueError('capture <= receipt <= decision is required')
    return {'version': VISION_VERSION, 'source': source, 'capture_s': capture_s,
            'received_s': received_s, 'decision_s': decision_s, 'time_s': time_s,
            'remaining_time': max(0., 28.-time_s), 'calibration_id': calibration_id,
            'ball': ball, 'opponent': opponent or {'position': (0., 0.), 'yaw': 0.,
                'position_valid': False, 'yaw_valid': False, 'age_s': 40., 'track_id': None},
            'sensors': sensors or {}}


class CameraVision(object):
    def __init__(self, calibration, source):
        if source not in ('naoqi_camera', 'simulated_camera'):
            raise ValueError('camera source required')
        self.calibration, self.source = calibration, source
        self.tracker = BallTracker()

    def reset(self):
        self.tracker.reset()

    def process(self, rgb, capture_s, received_s, decision_s, camera_to_ground,
                capture_odometry, current_odometry, time_s, sensors=None, synchronized=True):
        calibration = self.calibration
        measurement = None
        candidate = None
        if synchronized and rgb is not None:
            candidate = orange_candidate(rgb, calibration.width, calibration.height)
            if candidate is not None:
                measurement = calibration.project_ball(candidate, camera_to_ground)
        self.tracker.update(capture_s, received_s, measurement, capture_odometry)
        ball = self.tracker.estimate(decision_s, current_odometry)
        if not synchronized:
            ball['position_valid'] = ball['velocity_valid'] = ball['visible'] = False
            ball['reason'] = 'unsynchronized_sensors'
        state = vision_state(self.source, capture_s, received_s, decision_s, time_s, ball,
                             calibration_id=calibration.calibration_id, sensors=sensors)
        state['camera'] = {'candidate': candidate, 'synchronized': bool(synchronized)}
        return state


class ShooterSelector(object):
    """Local association, not a B-Human identity. Never infer yaw from velocity."""
    def __init__(self, max_age=.25, gate_m=.5):
        self.max_age, self.gate_m = max_age, gate_m
        self.reset()

    def reset(self):
        self.track_id, self.point, self.last_seen = 0, None, None

    def select(self, obstacles, ball_xy, now_s, odometry):
        candidates = [o for o in obstacles if not o.get('fallen', False) and
                      not o.get('type', '').startswith('fallen') and
                      o.get('type') not in ('teammate', 'ownPlayer', 'ownGoalkeeper') and
                      0 <= now_s-o['last_seen_s'] <= self.max_age and
                      math.hypot(o['position'][0]-ball_xy[0], o['position'][1]-ball_xy[1]) <= 1.2]
        if self.point is not None and self.last_seen is not None and now_s-self.last_seen <= self.max_age:
            previous = local(self.point, odometry)
            nearby = [o for o in candidates if math.hypot(o['position'][0]-previous[0],
                                                         o['position'][1]-previous[1]) <= self.gate_m]
            if nearby:
                candidates = nearby
        if not candidates:
            return None
        chosen = min(candidates, key=lambda o: math.hypot(o['position'][0]-ball_xy[0],
                                                         o['position'][1]-ball_xy[1]))
        point = ground(chosen['position'], odometry)
        if (self.point is None or now_s-self.last_seen > self.max_age or
                math.hypot(point[0]-self.point[0], point[1]-self.point[1]) > self.gate_m):
            self.track_id += 1
        self.point, self.last_seen = point, chosen['last_seen_s']
        return dict(chosen, position_valid=True, yaw=0., yaw_valid=False, track_id=self.track_id,
                    age_s=now_s-chosen['last_seen_s'])


class BHumanVisionAdapter(object):
    """Consumes explicit snapshots of local BallModel/ObstacleModel.

    Producer must export bhuman_local_models_v1 with clock-aligned seconds,
    positions mm, ball velocity mm/s, covariance mm^2, odometry m/rad.
    This adapter does not run the upstream perception providers.
    """
    def __init__(self):
        self.selector = ShooterSelector()
        self.last_capture = None

    def reset(self):
        self.selector.reset()
        self.last_capture = None

    def process(self, packet, received_s, decision_s, time_s, current_odometry=None):
        if packet['version'] != 'bhuman_local_models_v1' or packet['source'] not in ('bhuman_models', 'scene_privileged'):
            raise ValueError('explicit B-Human model schema/source required')
        capture, received_s, decision_s, time_s = vector((packet['capture_s'], received_s, decision_s, time_s), 4)
        if capture > received_s+1e-6 or received_s > decision_s+1e-6:
            raise ValueError('capture <= receipt <= decision is required')
        capture_pose = vector(packet['odometry'], 3)
        current_pose = capture_pose if current_odometry is None else vector(current_odometry, 3)
        model = packet['ball_model']
        last_seen = vector((model['last_seen_s'],), 1)[0]
        age = decision_s-last_seen
        new = self.last_capture is None or capture > self.last_capture
        if age < 0 or last_seen > capture:
            raise ValueError('invalid BallModel observation time')
        position = tuple(v*.001 for v in vector(model['position_mm'], 2))
        velocity = tuple(v*.001 for v in vector(model['velocity_mm_s'], 2))
        # BallModel.covariance describes position only. Never invent velocity
        # certainty when a producer has not supplied a validated uncertainty.
        velocity_cov = model.get('velocity_covariance_mm2_s2')
        velocity_valid = bool(model['velocity_valid'] and velocity_cov is not None)
        velocity_cov = covariance(velocity_cov, 1e-6) if velocity_cov is not None else ((0., 0.), (0., 0.))
        pose_delta = capture_pose[2]-current_pose[2]
        point_world = ground(position, capture_pose)
        velocity_world = ground(velocity, (0., 0., capture_pose[2]))
        latency = decision_s-capture
        if velocity_valid and age <= .25:
            point_world = tuple(point_world[i]+velocity_world[i]*latency for i in range(2))
        position_cov = covariance(model['covariance_mm2'], 1e-6)
        if velocity_valid:
            position_cov = tuple(tuple(position_cov[i][j]+latency**2*velocity_cov[i][j]
                                       for j in range(2)) for i in range(2))
        ball = {'position': local(point_world, current_pose),
                'velocity': local(velocity_world, (0., 0., current_pose[2])),
                'position_covariance': rotate_covariance(position_cov, pose_delta),
                'velocity_covariance': rotate_covariance(velocity_cov, pose_delta),
                'position_valid': bool(new and model['valid'] and age <= .25),
                'velocity_valid': bool(new and model['valid'] and velocity_valid and age <= .25),
                'visible': bool(new and model['seen_this_frame'] and age <= .25),
                'age_s': age, 'last_seen_s': last_seen, 'reason': 'bhuman_model' if new else 'duplicate'}
        obstacles = []
        for o in packet['obstacles']:
            obstacles.append({'position': local(ground(tuple(v*.001 for v in vector(o['center_mm'], 2)), capture_pose), current_pose),
                              'position_covariance': rotate_covariance(covariance(o['covariance_mm2'], 1e-6), pose_delta),
                              'last_seen_s': float(o['last_seen_s']), 'type': o['type'],
                              'fallen': bool(o.get('fallen', False))})
        opponent = (self.selector.select(obstacles, ball['position'], decision_s, current_pose)
                    if ball['position_valid'] else None)
        self.last_capture = max(capture, self.last_capture if self.last_capture is not None else capture)
        return vision_state(packet['source'], capture, received_s, decision_s, time_s, ball, opponent,
                            sensors=packet.get('sensors'))
