"""Load the B-Human NAO V6 model converted from its SimRobot RSI3 scene."""

import json
import math
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / 'models' / 'bhuman' / 'nao.xml'
SCENE = ROOT / 'sim' / 'scene.xml'
INITIAL_POSE = json.loads((ROOT / 'models' / 'bhuman' / 'initial_pose.json').read_text())
KINEMATIC_ROOT_Z = 0.334


def robot_spec(mocap=True):
    spec = mujoco.MjSpec.from_file(str(ROBOT))
    root = spec.worldbody.bodies[0]
    if mocap:
        root.mocap = True
    else:
        root.add_freejoint(name='root_free')
    # RobotDimensions.imuOffset is relative to the hip midpoint, 85 mm
    # below the RSI3 torso origin. Keep the original model file unchanged.
    root.add_site(name='imu', pos=[-0.008, 0.00606, 0.027], size=[0.003, 0, 0])
    for suffix, sensor_type in (('gyro', mujoco.mjtSensor.mjSENS_GYRO),
                                ('acc', mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
        spec.add_sensor(name='imu_' + suffix, type=sensor_type,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu')
    # Sensor-only additions; RSI3/model originals and physics remain unchanged.
    # Pinned B-Human RobotDimensions offsets (m) and nominal camera intrinsics.
    head = spec.body('Head')
    for name, x, z, tilt in (('upper', .05871, .06364, 1.2),
                             ('lower', .05071, .01774, 39.7)):
        angle = math.radians(tilt)
        # MuJoCo: local -z looks forward, +y is up, +x points image-right.
        head.add_camera(name='camera_'+name, pos=[x, 0., z], fovy=42.5,
                        xyaxes=[0., -1., 0., math.sin(angle), 0., math.cos(angle)])
    return spec


def scene_spec(mocap=True):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for name, x, yaw in (('striker', 0.3, math.pi), ('keeper', -1.08, 0.0)):
        frame = spec.worldbody.add_frame(pos=[x, 0, KINEMATIC_ROOT_Z if mocap else 0.4],
                                         quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        spec.attach(robot_spec(mocap=mocap), prefix=name + '_', frame=frame)
        coupling = spec.add_equality(name=name + '_hip_yaw_pitch_coupling')
        coupling.type = mujoco.mjtEq.mjEQ_JOINT
        coupling.name1 = name + '_LHipYawPitch'
        coupling.name2 = name + '_RHipYawPitch'
        coupling.data[0] = 0.0
        coupling.data[1] = 1.0
    return spec


def load_model(mocap=True):
    model = scene_spec(mocap=mocap).compile()
    # La geometría de contacto B-Human se superpone dentro de cada NAO.
    # Cada robot interactúa con cancha y balón, sin autocolisión ni choque
    # entre robots; los meshes visuales ya tienen contype=0.
    for geom_id in range(model.ngeom):
        if not model.geom_contype[geom_id]:
            continue
        body_name = model.body(model.geom_bodyid[geom_id]).name
        if body_name.startswith('striker_'):
            model.geom_contype[geom_id] = 2
            model.geom_conaffinity[geom_id] = 1
        elif body_name.startswith('keeper_'):
            model.geom_contype[geom_id] = 4
            model.geom_conaffinity[geom_id] = 1
        else:
            model.geom_conaffinity[geom_id] |= 6
    return model


def set_initial_pose(model, data):
    """Apply the RSI3 initial joint pose and matching position targets."""
    for role in ('striker', 'keeper'):
        for name, angle in INITIAL_POSE.items():
            joint = model.joint(role + '_' + name)
            data.qpos[joint.qposadr[0]] = angle
            if name != 'RHipYawPitch':
                data.ctrl[model.actuator(role + '_' + name + '_servo').id] = angle
