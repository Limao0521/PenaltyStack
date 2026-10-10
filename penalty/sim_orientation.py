"""Versioned RGB body orientation + own simulated encoders/IMU/odometry."""
import math

from penalty.physical_sim import PhysicalSimulator
from penalty.sim_opponent_vision import BHumanKeeperCamera
from penalty.sim_sensors import OwnSensorReader, load_sensor_model
from penalty.sim_vision import CameraDuelWorld, CameraKeeperController
from penalty.opponent_perception import merge_camera_results
from penalty.opponent_orientation import VisualYawEstimator

PIPELINE_VERSION = 'keeper_sensor_camera_v2'
STATE_VERSION = 'vision_state_v2'


class BodyCameraKeeperController(CameraKeeperController):
    """Frame ball and standing body together until an incoming shot."""
    def look(self, state):
        super().look(state)
        ball = state['ball']
        if not (ball['visible'] and ball['position_valid']):
            return
        distance = math.hypot(*ball['position'])
        incoming = ball['velocity_valid'] and ball['velocity'][0] < -.2
        if distance >= .65 and not incoming:
            height = state['sensors']['camera_to_ground']['lower'][11]
            self.tilt = max(.12, min(.4, .5*math.atan2(height-.0325, distance)))


class SensorCameraDuelWorld(CameraDuelWorld):
    def __init__(self, **options):
        super().__init__(simulator_factory=lambda: PhysicalSimulator(model_factory=load_sensor_model), **options)
        self.own_sensors = None

    def _install_controllers(self, rng, options):
        super()._install_controllers(rng, options)
        self.keeper = BodyCameraKeeperController()
        self.session.controllers['keeper'] = self.keeper

    def _ensure_session(self):
        session = super()._ensure_session()
        if self.own_sensors is None or self.own_sensors.sim is not session.sim:
            self.own_sensors = OwnSensorReader(session.sim)
        return session

    def _tick(self):
        super()._tick()
        self.own_sensors.update()


class SensorBHumanKeeperCamera(BHumanKeeperCamera):
    """Original box detector, new sensor transforms and visual-yaw contract."""
    def __init__(self, world, detector=None, orientation=None):
        super().__init__(world, detector)
        try:
            self.orientation = orientation or VisualYawEstimator()
        except Exception:
            self.close()
            raise

    def reset(self):
        super().reset()
        self.orientation.reset()

    def observe(self):
        world, sim = self.world, self.world.sim
        now = world.session.penalty_time
        sensors = world.own_sensors.read()
        synchronized = abs(float(sim.data.time)-sensors['capture_sim_s']) < 1e-6 and sensors['valid']
        pose = sensors['odometry']
        self.camera.renderer.update_scene(sim.data, camera='keeper_camera_lower')
        self.rgb = self.camera.renderer.render()
        self.camera.rgb = self.rgb
        self.upper_renderer.update_scene(sim.data, camera='keeper_camera_upper')
        self.upper_rgb = self.upper_renderer.render()
        transforms = sensors['camera_to_ground']
        state = self.camera.vision.process(self.rgb.tobytes(), now, now, now, transforms['lower'],
                                           pose, pose, now, sensors=sensors, synchronized=synchronized)
        world.keeper.look(state)
        detections = self.detector.detect(self.upper_rgb.tobytes(), 640, 480, now)
        lower_detections = self.detector.detect(self.rgb.tobytes(), 320, 240, now, camera='lower')
        upper = self.fusion.merge(state, detections, self.calibration, transforms['upper'],
                                  pose, pose, now, synchronized=synchronized)
        lower = self.lower_fusion.merge(state, lower_detections, self.lower_calibration,
                                        transforms['lower'], pose, pose, now, synchronized=synchronized)
        state = merge_camera_results(upper, lower)
        opponent = dict(state['opponent'])
        if opponent['position_valid']:
            selected = state['opponent_detector']['camera']
            t = transforms[selected]
            bearing = math.atan2(opponent['position'][1]-t[7], opponent['position'][0]-t[3])
            estimate = self.orientation.estimate(self.rgb if selected == 'lower' else self.upper_rgb,
                           opponent['box'], now, now, opponent['track_id'], bearing,
                           synchronized=synchronized, odometry_yaw=pose[2])
        else:
            self.orientation.reset()
            estimate = {'version': 'nao_visual_yaw_v1', 'yaw': 0., 'yaw_valid': False,
                        'yaw_variance': math.pi**2, 'capture_s': now, 'age_s': 0.,
                        'reason': 'opponent_position_missing', 'processing_ms': 0.}
        opponent.update(yaw=estimate['yaw'], yaw_valid=estimate['yaw_valid'],
                        yaw_variance=estimate['yaw_variance'], yaw_capture_s=estimate['capture_s'],
                        yaw_age_s=estimate['age_s'])
        return dict(state, version=STATE_VERSION, opponent=opponent, opponent_orientation=estimate,
                    camera_pipeline=PIPELINE_VERSION, runtime_provenance=self.detector.hello,
                    sensor_provenance=sensors['provenance'],
                    inference_clock='wall processing measured; simulated physics time not delayed')
