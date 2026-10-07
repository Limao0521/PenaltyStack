import importlib.util
import math
import unittest

import mujoco
import numpy as np


class PhysicalContractTests(unittest.TestCase):
    def require_adapter(self):
        self.assertIsNotNone(importlib.util.find_spec('penalty.physical_sim'),
                             'The physical MuJoCo sensor/actuator adapter is missing')
        from penalty.physical_sim import SensorAdapter
        from penalty.nao_model import load_model, set_initial_pose
        model = load_model(mocap=False)
        data = mujoco.MjData(model)
        set_initial_pose(model, data)
        mujoco.mj_forward(model, data)
        return SensorAdapter(model, data), model, data

    def test_two_free_robots_and_real_imu(self):
        adapter, model, data = self.require_adapter()
        self.assertEqual(model.nmocap, 0)
        self.assertEqual(model.nsensor, 4)
        frame = adapter.frame('keeper')
        self.assertEqual(len(frame['angles']), 26)
        self.assertEqual(len(frame['fsr']), 8)
        np.testing.assert_allclose(frame['gyro'], [0, 0, 0], atol=1e-8)
        # These free roots start in the air: specific acceleration is zero
        # in free fall, even though their instantaneous velocity is zero.
        np.testing.assert_allclose(frame['acc'], [0, 0, 0], atol=1e-6)

    def test_supported_torso_accelerometer_reports_positive_gravity(self):
        self.require_adapter()
        from penalty.physical_sim import SensorAdapter
        from penalty.nao_model import load_model, set_initial_pose
        model = load_model(mocap=False)
        data = mujoco.MjData(model)
        set_initial_pose(model, data)
        # Balance gravity at each body's centre of mass without constraining
        # the roots. A supported sensor reports +g in its local up axis.
        data.xfrc_applied[:, 2] = model.body_mass * 9.81
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(SensorAdapter(model, data).frame('keeper')['acc'],
                                   [0, 0, 9.81], atol=1e-6)

    def test_sensor_joint_names_and_shared_hip(self):
        adapter, model, data = self.require_adapter()
        address = model.joint('keeper_HeadYaw').qposadr[0]
        data.qpos[address] = 0.3
        hip = model.joint('keeper_LHipYawPitch').qposadr[0]
        data.qpos[hip] = 0.12
        mujoco.mj_forward(model, data)
        frame = adapter.frame('keeper')
        self.assertAlmostEqual(frame['angles'][0], 0.3)
        self.assertAlmostEqual(frame['angles'][14], 0.12)
        self.assertAlmostEqual(frame['angles'][20], 0.12)

    def test_stiffness_zero_disables_servo_and_invalid_output_is_atomic(self):
        adapter, model, data = self.require_adapter()
        angles = adapter.frame('keeper')['angles']
        adapter.apply('keeper', {'angles': angles, 'stiffness': [0] * 26})
        ids = adapter.actuator_ids['keeper']
        np.testing.assert_allclose(model.actuator_gainprm[ids, 0], 0)
        np.testing.assert_allclose(model.actuator_biasprm[ids, :3], 0)
        before = data.ctrl.copy()
        angles[0] = math.nan
        with self.assertRaises(ValueError):
            adapter.apply('keeper', {'angles': angles, 'stiffness': [100] * 26})
        np.testing.assert_equal(data.ctrl, before)


if __name__ == '__main__':
    unittest.main()
