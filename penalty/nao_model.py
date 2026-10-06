"""Construye en MuJoCo dos NAO H25 V6 a partir de su URDF.

El URDF contiene cinemática, límites, masas e inercias. Sus mallas Webots
`.mesh` no son compatibles con MuJoCo: se conservan el URDF original y sus
primitivas de colisión; las visuales de malla se omiten en la compilación.
"""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'models' / 'NAOH25V60.urdf'
SCENE = ROOT / 'sim' / 'scene.xml'

VISUALS = {
    'Torso_link': ('box', '0.10 0.12 0.19', '0 0 0.02'),
    'HeadYaw_link': ('cylinder', '0.028 0.04', '0 0 0.02'),
    'HeadPitch_link': ('sphere', '0.062', '0 0 0.055'),
}
for side in ('L', 'R'):
    VISUALS.update({
        side + 'ShoulderPitch_link': ('sphere', '0.034', '0 0 0'),
        side + 'ShoulderRoll_link': ('cylinder', '0.03 0.10', '0 0 -0.05'),
        side + 'ElbowYaw_link': ('sphere', '0.027', '0 0 0'),
        side + 'ElbowRoll_link': ('cylinder', '0.025 0.10', '0 0 -0.05'),
        side + 'WristYaw_link': ('sphere', '0.03', '0 0 0'),
        side + 'HipYawPitch_link': ('sphere', '0.04', '0 0 0'),
        side + 'HipPitch_link': ('cylinder', '0.033 0.10', '0 0 -0.05'),
        side + 'KneePitch_link': ('cylinder', '0.029 0.10', '0 0 -0.05'),
        side + 'AnklePitch_link': ('cylinder', '0.025 0.05', '0 0 -0.02'),
        side + 'AnkleRoll_link': ('box', '0.12 0.07 0.035', '0.03 0 -0.04'),
    })


def _primitive(link, tag, kind, size, xyz):
    element = ET.SubElement(link, tag)
    ET.SubElement(element, 'origin', xyz=xyz, rpy='0 0 0')
    geometry = ET.SubElement(element, 'geometry')
    values = size.split()
    if kind == 'box':
        ET.SubElement(geometry, 'box', size=size)
    elif kind == 'sphere':
        ET.SubElement(geometry, 'sphere', radius=values[0])
    else:
        ET.SubElement(geometry, 'cylinder', radius=values[0], length=values[1])
    if tag == 'visual':
        material = ET.SubElement(element, 'material', name='PenaltyNAOGrey')
        ET.SubElement(material, 'color', rgba='0.85 0.86 0.89 1')


def robot_spec(mocap=True):
    tree = ET.parse(str(URDF))
    robot = tree.getroot()
    links = {link.get('name'): link for link in robot.findall('link')}
    for link in robot.findall('link'):
        for visual in list(link.findall('visual')):
            if visual.find('.//mesh') is not None:
                link.remove(visual)
        description = VISUALS.get(link.get('name'))
        if description:
            if link.get('name') in ('LAnkleRoll_link', 'RAnkleRoll_link'):
                _primitive(link, 'collision', *description)
    # Los dedos y las dos manos carecen de inercia en este URDF. Para penales
    # no intervienen: se fijan sólo en el modelo derivado.
    for joint in robot.findall('joint'):
        if joint.get('type') == 'continuous' or joint.get('name') in ('LHand', 'RHand'):
            joint.set('type', 'fixed')
            limit = joint.find('limit')
            if limit is not None:
                joint.remove(limit)
    # Una articulación principal sin inercia impide compilar el modelo.
    for joint in robot.findall('joint'):
        if joint.get('type') not in ('revolute', 'continuous'):
            continue
        link = links[joint.find('child').get('link')]
        if link.find('inertial') is None:
            inertial = ET.SubElement(link, 'inertial')
            ET.SubElement(inertial, 'mass', value='0.001')
            ET.SubElement(inertial, 'inertia', ixx='0.000001', iyy='0.000001',
                          izz='0.000001', ixy='0', ixz='0', iyz='0')
    spec = mujoco.MjSpec.from_string(ET.tostring(robot, encoding='unicode'))
    for name, (kind, size, xyz) in VISUALS.items():
        geom_type = {'box': mujoco.mjtGeom.mjGEOM_BOX,
                     'sphere': mujoco.mjtGeom.mjGEOM_SPHERE,
                     'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER}[kind]
        values = [float(x) for x in size.split()]
        if kind == 'box':
            values = [x / 2 for x in values]
        elif kind == 'cylinder':
            values[1] /= 2
        values += [0.0] * (3 - len(values))
        spec.body(name).add_geom(name=name + '_visual', type=geom_type,
                                 size=values, pos=[float(x) for x in xyz.split()],
                                 rgba=[0.85, 0.86, 0.89, 1.0],
                                 contype=0, conaffinity=0)
    root = spec.worldbody.bodies[0]
    if mocap:
        root.mocap = True
    else:
        root.add_freejoint(name='root_free')
    return spec


def scene_spec(mocap=True):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for name, x, yaw in (('striker', 0.3, math.pi), ('keeper', -1.08, 0.0)):
        frame = spec.worldbody.add_frame(pos=[x, 0, 0.4],
                                         quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        spec.attach(robot_spec(mocap=mocap), prefix=name + '_', frame=frame)
        coupling = spec.add_equality(name=name + '_hip_yaw_pitch_coupling')
        coupling.type = mujoco.mjtEq.mjEQ_JOINT
        coupling.name1 = name + '_LHipYawPitch'
        coupling.name2 = name + '_RHipYawPitch'
        coupling.data[0] = 0.0
        coupling.data[1] = 1.0
        for joint in ET.parse(str(URDF)).getroot().findall('joint'):
            if joint.get('type') != 'revolute' or joint.get('name') in ('RHipYawPitch', 'LHand', 'RHand'):
                continue
            limit = joint.find('limit')
            actuator = spec.add_actuator(name=name + '_' + joint.get('name') + '_servo')
            actuator.target = name + '_' + joint.get('name')
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.set_to_position(kp=18, kv=1.5)
            actuator.ctrlrange = [float(limit.get('lower')), float(limit.get('upper'))]
            actuator.ctrllimited = True
            effort = float(limit.get('effort'))
            actuator.forcerange = [-effort, effort]
            actuator.forcelimited = True
    return spec


def load_model(mocap=True):
    model = scene_spec(mocap=mocap).compile()
    # Las primitivas Webots se solapan incluso en la postura inicial. Sus
    # contactos internos generan decenas de restricciones y aceleraciones
    # enormes; cada robot colisiona con cancha/balón, pero no consigo mismo.
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
        elif model.geom_contype[geom_id]:
            model.geom_conaffinity[geom_id] |= 6
    return model
