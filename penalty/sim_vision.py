"""Head-camera RGB -> vision_state_v1; no scene ball/opponent inputs.

Camera extrinsics, odometry and IMU are ideal *simulated sensors*. Only the
rendered pixels feed ball detection. Ground truth is allowed in validation
reports, never in this observation path. Not a real NAO calibration.
"""
import math

import mujoco
import numpy as np

from penalty.perception import CameraCalibration, CameraVision
from penalty.control import Command
from penalty.rl.controllers import TacticalKeeperController
from penalty.rl.duel import DuelWorld


class CameraKeeperController(TacticalKeeperController):
    """Head requests use perceived ball only; scene observations are ignored."""
    def __init__(self):
        super().__init__()
        self.pan, self.tilt = 0., .4

    def look(self, state):
        if state['ball']['visible'] and state['ball']['position_valid']:
            x, y = state['ball']['position']
            self.pan = max(-.5, min(.5, math.atan2(y, x)))
            # Camera-frame downward tilt, not the raw HeadPitch joint angle.
            self.tilt = max(.25, min(.8, math.atan2(.45, math.hypot(x, y))))

    def _keeper(self, obs):
        self._state('TRACK_BALL' if self.block is None else 'BLOCK', obs.now)
        command = Command(self.state, head_yaw=self.pan, block=self.block or 'center',
                          block_now=self.block is not None)
        command.head_pitch = self.tilt
        return command


class CameraDuelWorld(DuelWorld):
    def _install_controllers(self, rng, options):
        super()._install_controllers(rng, options)
        self.keeper = CameraKeeperController()
        self.session.controllers['keeper'] = self.keeper

    def _observe_keeper(self):
        # No scene-derived keeper observations even as a discarded return value.
        return None


class SimulatedKeeperCamera:
    def __init__(self, world, width=320, height=240, camera='lower'):
        self.world = world
        self.model = world.sim.model
        self.camera_name = 'keeper_camera_' + camera
        self.renderer = mujoco.Renderer(world.sim.model, height=height, width=width)
        camera_id = world.sim.model.camera(self.camera_name).id
        fovy = world.sim.model.cam_fovy[camera_id]
        focal = height/2/math.tan(math.radians(fovy)/2)
        self.vision = CameraVision(CameraCalibration(
            width, height, focal, focal, (width-1)/2, (height-1)/2, .0325,
            'mujoco_nominal_v1_' + camera), 'simulated_camera')
        self.rgb = None

    def reset(self):
        self.vision.reset()

    def close(self):
        self.renderer.close()

    def observe(self):
        world, sim = self.world, self.world.sim
        now = world.session.penalty_time
        xy, yaw = world._ground_pose('keeper')
        pose = (xy[0], xy[1], yaw)
        camera_id = sim.model.camera(self.camera_name).id
        self.renderer.update_scene(sim.data, camera=self.camera_name)
        self.rgb = self.renderer.render()
        # MuJoCo camera axes -> NAOqi optical convention (forward,left,up).
        camera_world = sim.data.cam_xmat[camera_id].reshape(3, 3) @ np.array(
            [[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
        c, s = math.cos(yaw), math.sin(yaw)
        world_to_ground = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        transform = np.eye(4)
        transform[:3, :3] = world_to_ground @ camera_world
        transform[:3, 3] = world_to_ground @ (sim.data.cam_xpos[camera_id] - np.array([*xy, 0.]))
        state = self.vision.process(self.rgb.tobytes(), now, now, now, transform.ravel(),
                                    pose, pose, now, sensors={
                                        'gyro': sim.data.sensor('keeper_imu_gyro').data.tolist()})
        state['sensor_provenance'] = 'ideal simulated camera extrinsics, IMU and feet odometry'
        world.keeper.look(state)
        return state
