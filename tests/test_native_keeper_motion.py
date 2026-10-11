"""Native protocol checks; measured MuJoCo initial sensors, no physics step."""
from copy import deepcopy
from dataclasses import asdict
import unittest

from penalty.bhuman_bridge import BHumanBridge
from penalty.keeper_defenses import DEFENSE_NAMES
from penalty.physical_sim import PhysicalSimulator
from penalty.skills.keeper_motion import MOTION_SKILL_VERSION, MotionParameters


def profile(parameters=MotionParameters()):
    return {'version': MOTION_SKILL_VERSION, 'parameters': asdict(parameters)}


class NativeKeeperMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Loading and reading the initial plant is permitted for sensor
        # validity. No mj_step, active root write or ball velocity write.
        sim = PhysicalSimulator(seed=80004)
        try:
            cls.sensors = sim.adapter.frame('keeper')
        finally:
            sim.close()

    def bridge(self, role='keeper'):
        bridge = BHumanBridge(role=role, trial='native-motion-contract', seed=80004)
        self.assertIn(MOTION_SKILL_VERSION, bridge.hello.get('capabilities', ()))
        return bridge

    def finish(self, *bridges):
        for bridge in bridges:
            bridge.close()
            self.assertIsNotNone(bridge.process.poll(), 'native process was not collected')

    def step(self, bridge, request, index):
        return bridge.step(deepcopy(self.sensors), request, sequence=index,
                           time_ms=10000+12*index)

    def reject(self, request, role='keeper', message=None):
        bridge = self.bridge(role)
        try:
            if message is None:
                with self.assertRaises(RuntimeError):
                    self.step(bridge, request, 0)
            else:
                with self.assertRaisesRegex(RuntimeError, message):
                    self.step(bridge, request, 0)
        finally:
            self.finish(bridge)

    def test_nominal_exactly_preserves_native_motor_outputs_for_all_six_defenses(self):
        for defense in DEFENSE_NAMES:
            with self.subTest(defense=defense):
                legacy, nominal = self.bridge(), self.bridge()
                try:
                    for index in range(64):
                        request = {'motion': 'stand'} if index < 4 or index >= 56 else {
                            'motion': 'block', 'block_side': defense}
                        original = self.step(legacy, dict(request), index)
                        extended_request = dict(request)
                        if request['motion'] == 'block':
                            extended_request['keeper_motion_skill'] = profile()
                        extended = self.step(nominal, extended_request, index)
                        for name in ('angles', 'stiffness', 'odometry', 'phase', 'busy',
                                     'kick_completed', 'motion_info', 'sensor_info'):
                            self.assertEqual(original[name], extended[name],
                                             '%s frame %d: %s changed' % (defense, index, name))
                        self.assertNotIn('keeper_motion_skill', original)
                        if index >= 4:
                            self.assertTrue(extended['keeper_motion_skill']['applied'])
                            self.assertEqual(extended['keeper_motion_skill']['changed_keyframes'], 0)
                            self.assertEqual(extended['keeper_motion_skill']['changed_angles'], 0)
                finally:
                    self.finish(legacy, nominal)

    def test_first_profile_reports_real_duration_and_joint_target_changes_then_compacts(self):
        params = MotionParameters(.9, 1.1, .08, .04)
        bridge = self.bridge()
        try:
            request = {'motion': 'block', 'block_side': 'center', 'keeper_motion_skill': profile(params)}
            first = self.step(bridge, request, 0)['keeper_motion_skill']
            self.assertEqual(first['version'], MOTION_SKILL_VERSION)
            self.assertTrue(first['applied'])
            self.assertGreater(first['changed_keyframes'], 0)
            self.assertGreater(first['changed_angles'], 0)
            self.assertEqual(first['clamped_angles'], 0)
            guard_blocks = {'genuflectFromStand', 'genuflectFromSitting', 'genuflect', 'genuflectWide'}
            changed_joint_names = set()
            frame_count = 0
            for block in first['blocks']:
                scale = params.guard_time_scale if block['id'] in guard_blocks else params.jump_time_scale
                for frame in block['keyframes']:
                    frame_count += 1
                    self.assertAlmostEqual(frame['duration_ms_applied'],
                                           frame['duration_ms_original']*scale, places=4)
                    self.assertGreater(frame['duration_ms_original'], 1.)
                    if block['id'] == 'jumpLeftFromSitting':
                        self.assertEqual(frame['index'], 1)
                    for angle in frame['angles']:
                        self.assertNotIn(angle['nominal_rad'], (10000., 20000.))
                        joint = angle['joint']
                        delta = (params.arm_spread_rad if joint == 'lShoulderRoll' else
                                 -params.arm_spread_rad if joint == 'rShoulderRoll' else
                                 params.guard_knee_offset_rad)
                        self.assertAlmostEqual(angle['target_rad']-angle['nominal_rad'], delta, places=6)
                        changed_joint_names.add(joint)
            self.assertGreater(frame_count, 0)
            self.assertEqual(changed_joint_names, {'lShoulderRoll', 'rShoulderRoll', 'lKneePitch', 'rKneePitch'})
            second = self.step(bridge, request, 1)['keeper_motion_skill']
            self.assertNotIn('blocks', second)
            self.assertEqual(second, {name: value for name, value in first.items() if name != 'blocks'})
            recovery = self.step(bridge, {'motion': 'getup'}, 2)['keeper_motion_skill']
            self.assertEqual(recovery, second)
        finally:
            self.finish(bridge)

    def test_native_rejects_out_of_bounds_parameters(self):
        for name, value in (('guard_time_scale', .84), ('guard_time_scale', 1.16),
                            ('jump_time_scale', .84), ('jump_time_scale', 1.16),
                            ('arm_spread_rad', -.081), ('arm_spread_rad', .121),
                            ('guard_knee_offset_rad', -.001), ('guard_knee_offset_rad', .101)):
            with self.subTest(parameter=name, value=value):
                invalid = profile()
                invalid['parameters'][name] = value
                self.reject({'motion': 'block', 'block_side': 'center', 'keeper_motion_skill': invalid})

    def test_native_rejects_bool_null_wrong_type_missing_and_unknown_fields(self):
        invalids = []
        for name in asdict(MotionParameters()):
            for value in (True, None, '1'):
                invalid = profile()
                invalid['parameters'][name] = value
                invalids.append(invalid)
        invalid = profile()
        del invalid['parameters']['arm_spread_rad']
        invalids.append(invalid)
        invalid = profile()
        invalid['parameters']['root_trajectory'] = 1.
        invalids.append(invalid)
        invalids.extend((None, [], {}, {'version': MOTION_SKILL_VERSION, 'parameters': None},
                         dict(profile(), extra=True), dict(profile(), version='keeper_motion_skill_v2')))
        for index, invalid in enumerate(invalids):
            with self.subTest(case=index):
                self.reject({'motion': 'block', 'block_side': 'center', 'keeper_motion_skill': invalid})

    def test_native_rejects_striker_preparation_and_non_block_profile_requests(self):
        self.reject({'motion': 'block', 'block_side': 'center', 'keeper_motion_skill': profile()},
                    role='striker', message='only valid for keeper block')
        self.reject({'motion': 'block', 'block_side': 'prepare', 'keeper_motion_skill': profile()},
                    message='preparation')
        for motion in ('stand', 'walk', 'getup', 'kick'):
            with self.subTest(motion=motion):
                self.reject({'motion': motion, 'keeper_motion_skill': profile()},
                            message='only valid for keeper block')

    def test_native_latches_profile_and_rejects_change_or_withdrawal(self):
        for changed in (None, profile(MotionParameters(.95, 1., 0., 0.))):
            with self.subTest(withdrawal=changed is None):
                bridge = self.bridge()
                try:
                    self.step(bridge, {'motion': 'block', 'block_side': 'center',
                                       'keeper_motion_skill': profile()}, 0)
                    request = {'motion': 'block', 'block_side': 'center'}
                    if changed is not None:
                        request['keeper_motion_skill'] = changed
                    with self.assertRaisesRegex(RuntimeError, 'cannot change or be omitted'):
                        self.step(bridge, request, 1)
                finally:
                    self.finish(bridge)

    def test_native_cannot_attach_profile_after_legacy_first_block(self):
        bridge = self.bridge()
        try:
            self.step(bridge, {'motion': 'block', 'block_side': 'center'}, 0)
            with self.assertRaisesRegex(RuntimeError, 'cannot change or be omitted'):
                self.step(bridge, {'motion': 'block', 'block_side': 'center',
                                   'keeper_motion_skill': profile()}, 1)
        finally:
            self.finish(bridge)


if __name__ == '__main__':
    unittest.main()
