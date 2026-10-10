"""MuJoCo head pixels -> actual B-Human robot inference + camera ball tracking.

New pipeline version; the published lower-camera-only benchmark is unchanged.
Transforms/odometry are ideal simulated own sensors, declared in each trace.
"""
import math

import mujoco
import numpy as np

from penalty.opponent_perception import (GroundCameraCalibration, OpponentFusion,
                                        ConfirmedLowerFusion, merge_camera_results)
from penalty.robot_detector import RobotDetector
from penalty.sim_vision import CameraDuelWorld, SimulatedKeeperCamera

PIPELINE_VERSION = 'keeper_bhuman_camera_v1'


class BHumanCameraDuelWorld(CameraDuelWorld):
    """Keep the validated ball-driven head and motion controllers."""


class BHumanKeeperCamera:
    def __init__(self, world, detector=None):
        self.world, self.model = world, world.sim.model
        self.camera = self.detector = self.upper_renderer = None
        self.owns_detector = detector is None
        try:
            self.camera = SimulatedKeeperCamera(world)
            self.upper_renderer = mujoco.Renderer(world.sim.model, height=480, width=640)
            self.detector = detector or RobotDetector()
            cam = world.sim.model.camera('keeper_camera_upper').id
            focal = 240/math.tan(math.radians(world.sim.model.cam_fovy[cam])/2)
            self.calibration = GroundCameraCalibration(640, 480, focal, focal, 319.5, 239.5,
                                                       'mujoco_nominal_upper_robot_v1')
            self.fusion = OpponentFusion()
            self.lower_fusion = ConfirmedLowerFusion()
            self.lower_fusion.selector = self.fusion.selector
            ball = self.camera.vision.calibration
            self.lower_calibration = GroundCameraCalibration(ball.width, ball.height, ball.fx, ball.fy,
                                                             ball.cx, ball.cy, 'mujoco_nominal_lower_robot_v1')
            self.rgb = self.upper_rgb = None
        except Exception:
            self.close()
            raise

    def reset(self):
        self.camera.reset()
        self.fusion.reset()
        self.lower_fusion.reset()

    def close(self):
        try:
            try:
                if self.camera is not None:
                    self.camera.close()
            finally:
                if self.upper_renderer is not None:
                    self.upper_renderer.close()
        finally:
            if self.detector is not None and self.owns_detector:
                self.detector.close()

    def observe(self):
        world, sim = self.world, self.world.sim
        state = self.camera.observe()
        self.rgb = self.camera.rgb
        now = world.session.penalty_time
        self.upper_renderer.update_scene(sim.data, camera='keeper_camera_upper')
        self.upper_rgb = self.upper_renderer.render()
        detections = self.detector.detect(self.upper_rgb.tobytes(), 640, 480, now)
        lower_detections = self.detector.detect(self.rgb.tobytes(), 320, 240, now, camera='lower')
        # Only own sensing/extrinsics, never striker pose or scene ball pose.
        xy, yaw = world._ground_pose('keeper')
        pose = (*xy, yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        to_ground = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        def transform(camera_name):
            cam = sim.model.camera(camera_name).id
            rotation = sim.data.cam_xmat[cam].reshape(3, 3) @ np.array(
                [[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
            t = np.eye(4)
            t[:3, :3] = to_ground @ rotation
            t[:3, 3] = to_ground @ (sim.data.cam_xpos[cam]-np.array([*xy, 0.]))
            return t.ravel()
        upper = self.fusion.merge(state, detections, self.calibration, transform('keeper_camera_upper'), pose, pose, now)
        lower = self.lower_fusion.merge(state, lower_detections, self.lower_calibration,
                                       transform('keeper_camera_lower'), pose, pose, now)
        state = merge_camera_results(upper, lower)
        state['camera_pipeline'] = PIPELINE_VERSION
        state['runtime_provenance'] = self.detector.hello
        state['sensor_provenance'] = ('ideal simulated own camera extrinsics/IMU/feet odometry; '
                                      'native inference wall time measured, simulation not delayed')
        return state
