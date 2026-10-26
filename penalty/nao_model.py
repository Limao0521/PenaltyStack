"""Load the B-Human NAO V6 model converted from its SimRobot RSI3 scene."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / 'models' / 'bhuman' / 'nao.xml'
SCENE = ROOT / 'sim' / 'scene.xml'
INITIAL_POSE = json.loads((ROOT / 'models' / 'bhuman' / 'initial_pose.json').read_text())
KINEMATIC_ROOT_Z = 0.334


CAMERA_FOVY_DEG = 42.5  # cameraIntrinsics.cfg openingAngleHeight (54.7° ancho en 4:3)
# RobotDimensions.cfg: offset desde la articulación del cuello y caída original.
CAMERAS = (
    ('CameraBottom', (0.05071, 0.0, 0.01774), 39.7),
    ('CameraTop', (0.05871, 0.0, 0.06364), 1.2),
)


def camera_quat(tilt_deg):
    """Orientación MuJoCo (mira por -Z, X a la derecha) para una caída frontal.

    Ejes de cámara en el marco del robot (X adelante, Y izquierda, Z arriba):
    derecha = -Y; la mira cae tilt_deg bajo la horizontal hacia +X. Las
    columnas de la matriz de rotación son los ejes de la cámara.
    """
    t = math.radians(tilt_deg)
    right = (0.0, -1.0, 0.0)
    up = (math.sin(t), 0.0, math.cos(t))
    back = (-math.cos(t), 0.0, math.sin(t))
    rotation = np.array((
        (right[0], up[0], back[0]),
        (right[1], up[1], back[1]),
        (right[2], up[2], back[2])), dtype=np.float64)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(9))
    return quat.tolist()


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
    # Cámaras de cabeza para las pruebas de visión del balón Trionda; van en
    # el cuerpo Head y cabecean con HeadPitch, como en el NAO real. Sólo
    # percepción de prueba: la FSM del ensayo sigue usando suelo verdadero.
    head = spec.body('Head')
    for name, pos, tilt_deg in CAMERAS:
        camera = head.add_camera(name=name)
        camera.pos = list(pos)
        camera.quat = camera_quat(tilt_deg)
        camera.fovy = CAMERA_FOVY_DEG
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
