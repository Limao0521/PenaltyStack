"""Physical NAO world and sensor/servo adapter, independent of session/UI."""
import math
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from penalty.motion_contract import (CYCLE_S, JOINT_NAMES, MODEL_JOINT_NAMES,
                                     SENSOR_OFF, SENSOR_IGNORE)
from penalty.nao_model import load_model, set_initial_pose

ROLES = ('striker', 'keeper')
# RobotDimensions.cfg: fl, fr, bl, br, metres in ankle/foot coordinates.
FSR_POSITIONS = (
    np.array([[.07025, .0299], [.07025, -.0231], [-.03025, .0299], [-.02965, -.0191]]),
    np.array([[.07025, .0231], [.07025, -.0299], [-.02965, .0191], [-.03025, -.0299]]),
)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class SensorAdapter:
    """Read real MuJoCo state; map 26 B-Human entries to 23 NAO servos."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.gain = model.actuator_gainprm.copy()
        self.bias = model.actuator_biasprm.copy()
        self.actuator_ids = {}
        self.joints = {}
        self.feet = {}
        for role in ROLES:
            self.joints[role] = [None if name.endswith('Hand') else
                                 model.joint(role + '_' + name)
                                 for name in MODEL_JOINT_NAMES]
            self.actuator_ids[role] = np.array([
                model.actuator(role + '_' + name + '_servo').id
                for name in MODEL_JOINT_NAMES
                if not name.endswith('Hand') and name != 'RHipYawPitch'])
            self.feet[role] = [model.body(role + '_' + side + 'Foot').id
                               for side in ('L', 'R')]
        self.floor_id = model.geom('floor').id

    def _fsr(self, role):
        loads = np.zeros((2, 4))
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geoms = (int(contact.geom[0]), int(contact.geom[1]))
            if self.floor_id not in geoms:
                continue
            other = geoms[1] if geoms[0] == self.floor_id else geoms[0]
            body = int(self.model.geom_bodyid[other])
            if body not in self.feet[role]:
                continue
            leg = self.feet[role].index(body)
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            # Contact frame row 0 is the normal. Include tangential force
            # when converting to a vertical load, independently of geom order.
            world_force = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
            kg = abs(float(world_force[2])) / 9.81
            rotation = self.data.xmat[body].reshape(3, 3)
            xy = (rotation.T @ (contact.pos - self.data.xpos[body]))[:2]
            positions = FSR_POSITIONS[leg]
            front = np.clip((xy[0] - positions[2:, 0].mean()) /
                            (positions[:2, 0].mean() - positions[2:, 0].mean()), 0, 1)
            left_front = np.clip((xy[1] - positions[1, 1]) /
                                 (positions[0, 1] - positions[1, 1]), 0, 1)
            left_back = np.clip((xy[1] - positions[3, 1]) /
                                (positions[2, 1] - positions[3, 1]), 0, 1)
            loads[leg] += kg * np.array([front * left_front, front * (1-left_front),
                                        (1-front)*left_back, (1-front)*(1-left_back)])
        return loads.ravel().tolist()

    def frame(self, role):
        if role not in ROLES:
            raise ValueError('unknown robot role')
        angles, velocity = [], []
        for index, joint in enumerate(self.joints[role]):
            if joint is None:
                angles.append(0.0)
                velocity.append(0.0)
            elif index == 20:  # one physical motor, one measured angle
                angles.append(angles[14])
                velocity.append(velocity[14])
            else:
                angles.append(float(self.data.qpos[joint.qposadr[0]]))
                velocity.append(float(self.data.qvel[joint.dofadr[0]]))
        body = self.model.body(role + '_Nao').id
        return {'angles': angles, 'velocity': velocity,
                'gyro': self.data.sensor(role + '_imu_gyro').data.copy().tolist(),
                'acc': self.data.sensor(role + '_imu_acc').data.copy().tolist(),
                'orientation': self.data.xquat[body].copy().tolist(),
                'fsr': self._fsr(role)}

    def apply(self, role, output):
        angles = np.asarray(output['angles'], dtype=float)
        stiffness = np.asarray(output['stiffness'], dtype=float)
        if (angles.shape != (26,) or stiffness.shape != (26,) or
                not np.isfinite(angles).all() or not np.isfinite(stiffness).all() or
                (stiffness < 0).any() or (stiffness > 100).any()):
            raise ValueError('B-Human output must contain 26 finite angles and stiffnesses 0..100')
        # Validate first; applying an invalid output must be atomic.
        for index, name in enumerate(MODEL_JOINT_NAMES):
            if name.endswith('Hand') or name == 'RHipYawPitch':
                continue
            actuator = self.model.actuator(role + '_' + name + '_servo').id
            target = angles[index]
            if target == SENSOR_IGNORE:
                # Ignore means preserve previous requested angle/rigidity.
                continue
            scale = 0 if target == SENSOR_OFF else stiffness[index] / 100.0
            self.model.actuator_gainprm[actuator] = self.gain[actuator] * scale
            self.model.actuator_biasprm[actuator] = self.bias[actuator] * scale
            if target != SENSOR_OFF:
                low, high = self.model.actuator_ctrlrange[actuator]
                self.data.ctrl[actuator] = np.clip(target, low, high)


class PhysicalSimulator:
    dt = CYCLE_S

    def __init__(self, bridge_factory=None, seed=0, ball_xy=(0.1, 0.0)):
        if (len(ball_xy) != 2 or not all(math.isfinite(v) for v in ball_xy) or
                abs(ball_xy[0]) > 1.2 or abs(ball_xy[1]) > 1.2):
            raise ValueError('ball_xy debe contener dos coordenadas finitas dentro de la cancha')
        if type(seed) is not int or not 0 <= seed <= 2**32-1:
            raise ValueError('seed debe ser un entero entre 0 y 2**32-1')
        self.model = load_model(mocap=False)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = 100
        self.data = mujoco.MjData(self.model)
        self.adapter = SensorAdapter(self.model, self.data)
        self.bridge_factory = bridge_factory
        self.bridges = {}
        self.mutation_lock = nullcontext
        self.seed = seed
        self.ball_xy = tuple(ball_xy)
        self.ball_qpos = self.model.joint('ball_free').qposadr[0]
        self.ball_qvel = self.model.joint('ball_free').dofadr[0]
        self.outputs = {role: {} for role in ROLES}
        self.trial = None
        self.kick_count = 0
        self.kick_contact_count = 0
        self.block_contacts = 0
        self.sequence = 0
        self.goal_event = None
        self.fallen = {role: False for role in ROLES}
        self._fall_time = {role: 0.0 for role in ROLES}
        self._had_foot_ball = False
        self.last_inputs = {}
        self.reset_world()

    def reset_world(self):
        with self.mutation_lock():
            self._reset_world()

    def _reset_world(self):
        mujoco.mj_resetData(self.model, self.data)
        self.model.actuator_gainprm[:] = self.adapter.gain
        self.model.actuator_biasprm[:] = self.adapter.bias
        set_initial_pose(self.model, self.data)
        for role, xy, yaw in (('striker', (.3, 0), math.pi),
                              ('keeper', (-1.08, 0), 0.0)):
            address = self.model.joint(role + '_root_free').qposadr[0]
            self.data.qpos[address:address+7] = [xy[0], xy[1], .334,
                math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        self.data.qpos[self.ball_qpos:self.ball_qpos+7] = [*self.ball_xy, .0325, 1, 0, 0, 0]
        self.sequence = self.kick_count = self.kick_contact_count = self.block_contacts = 0
        self.goal_event = None
        self.outputs = {role: {} for role in ROLES}
        self.fallen = {role: False for role in ROLES}
        self._fall_time = {role: 0.0 for role in ROLES}
        self._had_foot_ball = False
        self.last_inputs = {}
        mujoco.mj_forward(self.model, self.data)

    def reset(self, trial):
        self.close()
        self.trial = trial
        self.reset_world()
        if self.bridge_factory is None:
            from penalty.bhuman_bridge import BHumanBridge
            factory = BHumanBridge
        else:
            factory = self.bridge_factory
        try:
            for role in ROLES:
                self.bridges[role] = factory(role=role, trial=trial, seed=self.seed,
                    ball_radius_mm=float(self.model.geom('orange_ball').size[0])*1000)
        except Exception:
            self.close()
            raise
        return self.observation()

    def pose(self, role):
        body = self.model.body(role + '_Nao').id
        rot = self.data.xmat[body].reshape(3, 3)
        xyz = self.data.xpos[body].copy()
        return xyz, math.atan2(rot[1, 0], rot[0, 0])

    def observation(self):
        bx, by = self.data.qpos[self.ball_qpos:self.ball_qpos+2]
        ball_speed = float(np.linalg.norm(self.data.qvel[self.ball_qvel:self.ball_qvel+2]))
        values = {}
        for role in ROLES:
            xyz, yaw = self.pose(role)
            # B-Human's RobotPose uses the midpoint of the two feet, as in
            # SimulatedRobot3D::getRobotPose. The RSI3 torso origin is not
            # that frame; its offset changes while the robot leans/walks.
            feet = self.adapter.feet[role]
            ground_xy = (self.data.xpos[feet[0],:2]+self.data.xpos[feet[1],:2])*.5
            goal = (-1.2, 0) if role == 'striker' else (.1, 0)
            dx, dy = bx-ground_xy[0], by-ground_xy[1]
            cos, sin = math.cos(yaw), math.sin(yaw)
            output = self.outputs[role]
            values[role] = {'ball_range': math.hypot(dx, dy),
                'ball_bearing': wrap(math.atan2(dy, dx)-yaw),
                'goal_bearing': wrap(math.atan2(goal[1]-ground_xy[1], goal[0]-ground_xy[0])-yaw),
                'ball_speed': ball_speed,
                'ball': [(cos*dx+sin*dy)*1000, (-sin*dx+cos*dy)*1000],
                'motion_busy': bool(output.get('busy', False)),
                'motion_done': bool(output.get('kick_completed', False)),
                'phase': output.get('phase', 'not_started')}
        return values

    def _events(self):
        ball_geom = self.model.geom('orange_ball').id
        ball_x = float(self.data.qpos[self.ball_qpos])
        if self.goal_event is None and self._previous_ball[0] >= -1.2 > ball_x:
            p = self.data.qpos[self.ball_qpos:self.ball_qpos+3]
            fraction = (-1.2-self._previous_ball[0]) / (p[0]-self._previous_ball[0])
            crossing = self._previous_ball + fraction*(p-self._previous_ball)
            if abs(crossing[1]) < .75-.0325 and crossing[2] < .8-.0325:
                self.goal_event = 'goal'
            else:
                self.goal_event = 'wide'
        self._previous_ball = self.data.qpos[self.ball_qpos:self.ball_qpos+3].copy()
        foot_contact = False
        for c in self.data.contact:
            a, b = map(int, c.geom)
            if ball_geom not in (a, b):
                continue
            other = b if a == ball_geom else a
            body = self.model.body(self.model.geom_bodyid[other]).name
            if body in ('striker_LFoot', 'striker_RFoot'):
                foot_contact = True
            if body.startswith('keeper_'):
                self.block_contacts += 1
        if foot_contact and not self._had_foot_ball:
            self.kick_contact_count += 1
        self._had_foot_ball = foot_contact
        for role in ROLES:
            body = self.model.body(role+'_Nao').id
            tilt = math.acos(np.clip(self.data.xmat[body].reshape(3,3)[2,2], -1, 1))
            self._fall_time[role] = self._fall_time[role]+self.model.opt.timestep if tilt > math.pi/4 else 0
            torso_contact = any(self.model.geom_bodyid[c.geom[0]] == body or
                                self.model.geom_bodyid[c.geom[1]] == body
                                for c in self.data.contact if self.adapter.floor_id in c.geom)
            if self._fall_time[role] >= .2 or torso_contact:
                self.fallen[role] = True

    def step(self, requests):
        if len(self.bridges) != 2:
            raise RuntimeError('B-Human is not initialized for both robots')
        observations = self.observation()
        pending = {}
        self.last_inputs = {}
        for role in ROLES:
            request = dict(requests.get(role, {'motion': 'stand'}))
            request['ball'] = observations[role]['ball']
            sensors = self.adapter.frame(role)
            self.last_inputs[role] = {'sensors': sensors, 'request': request}
            pending[role] = self.bridges[role].step(sensors, request,
                sequence=self.sequence, time_ms=10000+round(self.data.time*1000))
        # Native pipe waits above never hold the viewer's mutex.
        with self.mutation_lock():
            self._apply_and_step(pending)
        self.sequence += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('non-finite MuJoCo state')
        return self.observation()

    def _apply_and_step(self, pending):
        # Validate each output before mutating the world or stepping physics.
        for role, output in pending.items():
            self.adapter.apply(role, output)
            previous = self.outputs[role]
            if role == 'striker' and output.get('phase') == 'kick' and previous.get('phase') != 'kick':
                self.kick_count += 1
            self.outputs[role] = output
        self._previous_ball = self.data.qpos[self.ball_qpos:self.ball_qpos+3].copy()
        for _ in range(6):
            mujoco.mj_step(self.model, self.data)
            self._events()
        # mj_step integrates after computing sensors. Recompute once at the
        # control boundary so joint angles, IMU and FSR share the timestamp.
        mujoco.mj_forward(self.model, self.data)

    def stable(self):
        for role in ROLES:
            body = self.model.body(role+'_Nao').id
            tilt = math.acos(np.clip(self.data.xmat[body].reshape(3,3)[2,2], -1, 1))
            frame = self.adapter.frame(role)
            if tilt > math.radians(10) or np.linalg.norm(frame['gyro']) >= .2:
                return False
            if sum(frame['fsr'][:4]) <= .05 or sum(frame['fsr'][4:]) <= .05:
                return False
        return True

    def result(self):
        if self.goal_event:
            return self.goal_event
        p = self.data.qpos[self.ball_qpos:self.ball_qpos+3]
        if abs(p[1]) > 1.2 or p[0] > 1.2:
            return 'out'
        if self.block_contacts and np.linalg.norm(self.data.qvel[self.ball_qvel:self.ball_qvel+3]) < .1:
            return 'blocked'
        return 'in_play'

    def snapshot(self):
        return {'time_s': float(self.data.time), 'result': self.result(),
            'kick_count': self.kick_count, 'kick_contact_count': self.kick_contact_count,
            'block_contacts': self.block_contacts,
            'ball_position': self.data.qpos[self.ball_qpos:self.ball_qpos+3].tolist(),
            'robots': {role: {'phase': self.outputs[role].get('phase', 'not_started'),
                'motion': self.outputs[role].get('motion', 'stand'),
                'fall_state': self.outputs[role].get('sensor_info',{}).get('fall_state','not_started'),
                'fallen': self.fallen[role], 'fsr': self.adapter.frame(role)['fsr'],
                'position': self.pose(role)[0].tolist()} for role in ROLES}}

    def trace(self):
        return {'time_s': float(self.data.time), 'sequence': self.sequence,
                'snapshot': self.snapshot(), 'sensors': {r:self.adapter.frame(r) for r in ROLES},
                'inputs': self.last_inputs, 'outputs': self.outputs,
                'actuator_force': self.data.actuator_force.tolist()}

    def close(self):
        for bridge in self.bridges.values():
            bridge.close()
        self.bridges.clear()
