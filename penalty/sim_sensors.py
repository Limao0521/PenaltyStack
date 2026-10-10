"""Own encoders/IMU/FSR -> camera kinematics and contact odometry (v1).

The vision path reads MuJoCo sensordata, not torso/feet/camera world poses.
Gyro yaw is relative, with no absolute scene compass. Contact odometry can
drift/slip; gravity correction and nominal sole geometry are approximations.
"""
import math

import mujoco
import numpy as np

from penalty.motion_contract import MODEL_JOINT_NAMES
from penalty.nao_model import scene_spec, robot_spec

VERSION = 'simulated_own_sensors_v1'
ENCODERS = tuple(name for name in MODEL_JOINT_NAMES if not name.endswith('Hand'))
SOLE = np.array([[-.055, -.0345, -.04511], [-.055, .0405, -.04511],
                 [.095, -.0345, -.04511], [.095, .0405, -.04511]])


def load_sensor_model(mocap=False):
    """New sensor-only scene variant; original MJCF and default stay intact."""
    spec = scene_spec(mocap=mocap)
    for name in ENCODERS:
        spec.add_sensor(name='keeper_encoder_'+name, type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname='keeper_'+name)
    model = spec.compile()
    # Preserve the same contact exclusions as load_model.
    for geom in range(model.ngeom):
        if not model.geom_contype[geom]:
            continue
        body = model.body(model.geom_bodyid[geom]).name
        if body.startswith('striker_'):
            model.geom_contype[geom], model.geom_conaffinity[geom] = 2, 1
        elif body.startswith('keeper_'):
            model.geom_contype[geom], model.geom_conaffinity[geom] = 4, 1
        else:
            model.geom_conaffinity[geom] |= 6
    return model


def rotation(roll, pitch, yaw=0.):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                            math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


class OwnSensorReader:
    """Reads keeper sensors only; FK runs in isolated robot-local data."""
    def __init__(self, simulator):
        self.sim = simulator
        self.fk_model = robot_spec(mocap=False).compile()
        self.fk = mujoco.MjData(self.fk_model)
        self.joints = [(name, self.fk_model.joint(name).qposadr[0]) for name in ENCODERS]
        self.feet = [self.fk_model.body(s+'Foot').id for s in ('L', 'R')]
        self.root = self.fk_model.joint('root_free').qposadr[0]
        self.reset()

    def reset(self):
        self.last_time = None
        self.attitude = np.eye(3)
        self.torso_xy = np.zeros(2)
        self.previous_feet = self.previous_load = None
        self.packet = None

    def update(self):
        sim, fk = self.sim, self.fk
        now = float(sim.data.time)
        if self.last_time is not None and now < self.last_time:
            self.reset()
        dt = 0. if self.last_time is None else now-self.last_time
        if self.last_time == now:
            return
        angles = {name: float(sim.data.sensor('keeper_encoder_'+name).data[0]) for name in ENCODERS}
        gyro = sim.data.sensor('keeper_imu_gyro').data.copy()
        acc = sim.data.sensor('keeper_imu_acc').data.copy()
        fsr = np.asarray(sim.adapter._fsr('keeper')).reshape(2, 4)
        if not np.isfinite([*angles.values(), *gyro, *acc, *fsr.ravel()]).all():
            self.packet = None
            self.last_time = now
            return
        # Body-frame gyro is integrated at every 12 ms physics/control tick.
        speed = float(np.linalg.norm(gyro))
        if dt > 0 and speed > 1e-10:
            axis = gyro/speed
            x, y, z = axis
            skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            angle = min(dt, .05)*speed
            self.attitude = self.attitude @ (np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew))
        roll = math.atan2(self.attitude[2, 1], self.attitude[2, 2])
        pitch = math.atan2(-self.attitude[2, 0], math.hypot(self.attitude[2, 1], self.attitude[2, 2]))
        yaw = math.atan2(self.attitude[1, 0], self.attitude[0, 0])
        norm = float(np.linalg.norm(acc))
        # Accelerometer is also affected by motion. Correct gravity slowly
        # while supported; never use the scene's quaternion/absolute yaw.
        if 8.5 <= norm <= 11.2 and fsr.sum() > 1.:
            a_roll = math.atan2(acc[1], acc[2])
            a_pitch = math.atan2(-acc[0], math.hypot(acc[1], acc[2]))
            gain = 1. if self.last_time is None else 1-math.exp(-dt/1.5)
            roll += gain*((a_roll-roll+math.pi) % (2*math.pi)-math.pi)
            pitch += gain*(a_pitch-pitch)
        self.attitude = rotation(roll, pitch, yaw)
        rp = rotation(roll, pitch)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rp.ravel())
        fk.qpos[self.root:self.root+7] = [0., 0., 0., *quat]
        for name, address in self.joints:
            fk.qpos[address] = angles[name]
        mujoco.mj_kinematics(self.fk_model, fk)
        mujoco.mj_camlight(self.fk_model, fk)
        feet = fk.xpos[self.feet].copy()
        c, s = math.cos(yaw), math.sin(yaw)
        rz = np.array([[c, -s], [s, c]])
        horizontal = feet[:, :2] @ rz.T
        load = fsr.sum(1)
        if self.previous_feet is None:
            self.torso_xy = -horizontal.mean(0)
        else:
            common = np.minimum(load, self.previous_load)
            if common.sum() > .5:
                common /= common.sum()
                self.torso_xy += (common[:, None]*(self.previous_feet-horizontal)).sum(0)
        ground_xy = self.torso_xy+horizontal.mean(0)
        self.previous_feet, self.previous_load = horizontal, load
        origin = feet.mean(0)
        soles = [fk.xpos[f]+SOLE@fk.xmat[f].reshape(3, 3).T for f in self.feet]
        supported = [soles[i][:, 2].min() for i in range(2) if load[i] > .2]
        origin[2] = min(supported) if supported else min(p[:, 2].min() for p in soles)
        transforms = {}
        for name in ('upper', 'lower'):
            cam = self.fk_model.camera('camera_'+name).id
            t = np.eye(4)
            # MuJoCo -z optical forward -> NAO (forward,left,up).
            t[:3, :3] = fk.cam_xmat[cam].reshape(3, 3) @ np.array(
                [[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
            t[:3, 3] = fk.cam_xpos[cam]-origin
            transforms[name] = t.ravel().tolist()
        self.packet = {'version': VERSION, 'capture_sim_s': now, 'encoders': angles,
                       'gyro': gyro.tolist(), 'acc': acc.tolist(), 'fsr_kg': fsr.ravel().tolist(),
                       'torso_roll': roll, 'torso_pitch': pitch, 'relative_yaw': yaw,
                       'odometry': [float(ground_xy[0]), float(ground_xy[1]), yaw],
                       'camera_to_ground': transforms,
                       'valid': bool(dt <= .05 and load.sum() > .5 and abs(roll) < .35 and abs(pitch) < .35),
                       'provenance': 'MuJoCo jointpos/gyro/accelerometer sensors; contact-derived own FSR; '
                                     'encoder FK, gyro/gravity attitude and support-foot odometry; '
                                     'nominal geometry, ideal sensor readings, drift/slip possible; no world pose sensors'}
        self.last_time = now

    def read(self):
        self.update()
        if self.packet is None:
            raise RuntimeError('own camera sensor frame unavailable')
        return self.packet
