"""Interfaz de acciones articulares para experimentos futuros de RL.

No incluye política, recompensa ni marcha estable. Cada acción contiene 23
objetivos angulares por NAO; los dos HipYawPitch se acoplan como en hardware.
Manos y dedos se fijan en el modelo derivado porque el URDF no les da inercia.
"""
import math

import mujoco
import numpy as np

from penalty.nao_model import load_model


class JointSpaceEnv:
    def __init__(self):
        self.model = load_model(mocap=False)
        self.data = mujoco.MjData(self.model)
        self.action_names = tuple(self.model.actuator(i).name for i in range(self.model.nu))
        self.low = self.model.actuator_ctrlrange[:, 0].copy()
        self.high = self.model.actuator_ctrlrange[:, 1].copy()
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for role, xyz, yaw in (
            ('striker', (0.3, 0.0, 0.4), math.pi),
            ('keeper', (-1.08, 0.0, 0.4), 0.0),
        ):
            address = self.model.joint(role + '_root_free').qposadr[0]
            self.data.qpos[address:address + 7] = [xyz[0], xyz[1], xyz[2],
                                                    math.cos(yaw / 2), 0, 0,
                                                    math.sin(yaw / 2)]
        ball = self.model.joint('ball_free').qposadr[0]
        self.data.qpos[ball:ball + 7] = [0.1, 0, 0.0325, 1, 0, 0, 0]
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self):
        return {'qpos': self.data.qpos.copy(), 'qvel': self.data.qvel.copy(),
                'time_s': float(self.data.time)}

    def step(self, action, repeats=10):
        values = np.asarray(action, dtype=float)
        if values.shape != (self.model.nu,) or not np.isfinite(values).all():
            raise ValueError('action must have %d finite joint targets' % self.model.nu)
        self.data.ctrl[:] = np.clip(values, self.low, self.high)
        for _ in range(repeats):
            mujoco.mj_step(self.model, self.data)
        return self.observation()
