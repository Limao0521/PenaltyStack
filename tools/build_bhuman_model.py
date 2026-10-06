"""Convert the pinned B-Human SimRobot NAO V6 description to MuJoCo assets.

The authoritative RSI3 and license stay unchanged. This exporter uses the
same meshes, joint frames, collision primitives, masses and servo parameters
that SimRobotCore3 reads; the generated MJCF is a local MuJoCo adaptation.
"""

import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'models' / 'bhuman' / 'NaoV6H25.rsi3'
DEST = ROOT / 'models' / 'bhuman'
MESH_DIR = DEST / 'meshes'
NUMBER = re.compile(r'^([-+\d.eE]+)(mm|g|degree|g\*mm\^2)?$')
RGB = re.compile(r'rgb\((\d+)%\s*,\s*(\d+)%\s*,\s*(\d+)%\)')


def unit(value):
    match = NUMBER.match(value)
    if not match:
        raise ValueError('unsupported RSI3 unit: ' + value)
    number, suffix = match.groups()
    return float(number) * {'mm': .001, 'g': .001, 'degree': math.pi / 180,
                            'g*mm^2': 1e-9, None: 1}[suffix]


def vec(element, default=0):
    return [unit(element.get(axis, str(default))) if element is not None else default
            for axis in 'xyz']


def fmt(values):
    return ' '.join(format(v, '.10g') for v in values)


def numbers(text, kind=float):
    return [kind(token) for line in (text or '').splitlines()
            for token in line.split('#', 1)[0].split()]


def appearance_refs(element, definitions):
    name = element.get('ref')
    if not name:
        return []
    if element.tag == 'ComplexAppearance':
        return [name]
    definition = definitions['Appearance'][name]
    return [part for child in definition if child.tag in ('Appearance', 'ComplexAppearance')
            for part in appearance_refs(child, definitions)]


def build():
    root = ET.parse(SOURCE).getroot()
    definitions = {
        tag: {item.get('name'): item for item in root.iter(tag) if item.get('name')}
        for tag in ('Appearance', 'ComplexAppearance', 'Vertices', 'Normals', 'Surface')
    }
    robot = root.find('Body')
    assert robot.get('name') == 'Nao'
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    used_meshes = {
        name for body in robot.iter('Body')
        for appearance in body if appearance.tag in ('Appearance', 'ComplexAppearance')
        for name in appearance_refs(appearance, definitions)
    }
    for unused in definitions['ComplexAppearance'].keys() - used_meshes:
        (MESH_DIR / (unused + '.obj')).unlink(missing_ok=True)

    # RSI3 stores triangle indices and shared vertex arrays directly in XML.
    for name in sorted(used_meshes):
        part = definitions['ComplexAppearance'][name]
        vertex_element = part.find('Vertices')
        if vertex_element.get('ref'):
            vertex_element = definitions['Vertices'][vertex_element.get('ref')]
        vertices = numbers(vertex_element.text)
        raw_indices = numbers(part.find('Triangles').text, int)
        # SimRobot interleaves vertex and normal indices when Normals exists.
        normal_element = part.find('Normals')
        stride = 2 if normal_element is not None else 1
        triangles = raw_indices[::stride]
        if len(raw_indices) % (3 * stride):
            raise ValueError('malformed RSI3 triangles: ' + name)
        if len(vertices) % 3 or len(triangles) % 3 or max(triangles) >= len(vertices) // 3:
            raise ValueError('malformed RSI3 mesh: ' + name)
        lines = ['v %s %s %s' % tuple(vertices[i:i + 3])
                 for i in range(0, len(vertices), 3)]
        if normal_element is not None:
            if normal_element.get('ref'):
                normal_element = definitions['Normals'][normal_element.get('ref')]
            normals = numbers(normal_element.text)
            if len(normals) % 3 or max(raw_indices[1::2]) >= len(normals) // 3:
                raise ValueError('malformed RSI3 normals: ' + name)
            lines += ['vn %s %s %s' % tuple(normals[i:i + 3])
                      for i in range(0, len(normals), 3)]
            lines += ['f ' + ' '.join('%d//%d' % (raw_indices[k + 2 * j] + 1,
                                                 raw_indices[k + 2 * j + 1] + 1)
                                     for j in range(3))
                      for k in range(0, len(raw_indices), 6)]
        else:
            lines += ['f %d %d %d' % tuple(index + 1 for index in triangles[i:i + 3])
                      for i in range(0, len(triangles), 3)]
        (MESH_DIR / (name + '.obj')).write_text('\n'.join(lines) + '\n', encoding='ascii')

    model = ET.Element('mujoco', model='bhuman_nao_v6')
    ET.SubElement(model, 'compiler', angle='radian', inertiafromgeom='false')
    assets = ET.SubElement(model, 'asset')
    for name in sorted(used_meshes):
        ET.SubElement(assets, 'mesh', name=name, file='meshes/' + name + '.obj')
    world = ET.SubElement(model, 'worldbody')
    body_root = ET.SubElement(world, 'body', name='Nao')
    joints = []
    initial = {}

    def add_body(source_body, target_body):
        inertia = source_body.find('InertiaMatrixMass')
        if inertia is not None:
            ET.SubElement(target_body, 'inertial',
                          mass=fmt([unit(inertia.get('value'))]),
                          pos=fmt(vec(inertia)),
                          fullinertia=fmt([unit(inertia.get(x))
                                           for x in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')]))
        for geom in source_body:
            if geom.tag not in ('BoxGeometry', 'SphereGeometry'):
                continue
            attributes = {'type': 'box' if geom.tag == 'BoxGeometry' else 'sphere',
                          'pos': fmt(vec(geom.find('Translation'))),
                          'rgba': '0 0 0 0', 'group': '3'}
            if geom.tag == 'BoxGeometry':
                attributes['size'] = fmt([unit(geom.get(x)) / 2
                                          for x in ('depth', 'width', 'height')])
            else:
                attributes['size'] = fmt([unit(geom.get('radius'))])
            rotation = geom.find('Rotation')
            if rotation is not None:
                attributes['euler'] = fmt(vec(rotation))
            ET.SubElement(target_body, 'geom', **attributes)
        for appearance in source_body:
            if appearance.tag not in ('Appearance', 'ComplexAppearance'):
                continue
            for name in appearance_refs(appearance, definitions):
                part = definitions['ComplexAppearance'][name]
                surface = part.find('Surface').get('ref').replace('$TeamColor', 'skin')
                color = definitions['Surface'].get(surface)
                match = RGB.search(color.get('diffuseColor', '')) if color is not None else None
                rgba = [int(x) / 100 for x in match.groups()] + [1] if match else [.8, .8, .8, 1]
                if surface in ('NaoMat_MainSkin', 'nao-skin'):
                    rgba = [.92, .94, .97, 1]
                ET.SubElement(target_body, 'geom', name=source_body.get('name') + '_' + name,
                              type='mesh', mesh=name,
                              rgba=fmt(rgba), contype='0', conaffinity='0', group='1')
        for hinge in source_body.findall('Hinge'):
            child = hinge.find('Body')
            name = hinge.get('name')
            body = ET.SubElement(target_body, 'body', name=child.get('name'),
                                 pos=fmt(vec(hinge.find('Translation'))))
            axis = hinge.find('Axis')
            deflection = axis.find('Deflection')
            low = unit(deflection.get('min'))
            high = unit(deflection.get('max'))
            init = unit(deflection.get('init')) if deflection.get('init') else min(high, max(low, 0))
            ET.SubElement(body, 'joint', name=name, type='hinge',
                          axis=fmt(vec(axis)), range=fmt([low, high]), limited='true')
            servo = axis.find('ServoMotor')
            joints.append((name, low, high, servo))
            initial[name] = init
            add_body(child, body)

    add_body(robot, body_root)
    actuators = ET.SubElement(model, 'actuator')
    for name, low, high, servo in joints:
        if name == 'RHipYawPitch':
            continue
        force = unit(servo.get('maxForce'))
        ET.SubElement(actuators, 'position', name=name + '_servo', joint=name,
                      kp=fmt([unit(servo.get('p'))]), kv=fmt([unit(servo.get('d'))]),
                      ctrlrange=fmt([low, high]), ctrllimited='true',
                      forcerange=fmt([-force, force]), forcelimited='true')
    tree = ET.ElementTree(model)
    ET.indent(tree, space='  ')
    tree.write(DEST / 'nao.xml', encoding='utf-8', xml_declaration=True)
    (DEST / 'initial_pose.json').write_text(json.dumps(initial, sort_keys=True, indent=2) + '\n',
                                            encoding='utf-8')
    return len(joints), len(used_meshes)


if __name__ == '__main__':
    print('joints=%d meshes=%d' % build())
