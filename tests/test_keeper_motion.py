"""Contracts and request isolation for bounded keeper keyframe training."""
from dataclasses import asdict
import math
import unittest
from unittest.mock import Mock, patch

from penalty.rl.motion_world import MotionDuelWorld, MotionPhysicalSimulator
from penalty.skills.keeper_motion import (MOTION_SKILL_VERSION, MotionParameters,
                                         ParameterizedKeeperMotion, motion_parameters)


class KeeperMotionContractTests(unittest.TestCase):
    def test_all_four_parameters_are_real_runtime_supported_fields(self):
        skill = ParameterizedKeeperMotion()
        self.assertEqual(tuple(skill.bounds()), tuple(asdict(MotionParameters())))
        self.assertEqual(skill.RUNTIME_SUPPORTED, tuple(skill.bounds()))
        params = MotionParameters(.9, 1.1, .08, .04)
        self.assertEqual(skill.apply(params), {'keeper_motion_skill': {
            'version': MOTION_SKILL_VERSION, 'parameters': asdict(params)}})
        self.assertEqual(skill.record()['parameters'], asdict(params))

    def test_exact_nominal_and_vector_mapping(self):
        skill = ParameterizedKeeperMotion()
        self.assertEqual(skill.current, MotionParameters(1., 1., 0., 0.))
        self.assertEqual(skill.from_vector([.9, 1., .02, .03]), MotionParameters(.9, 1., .02, .03))
        self.assertEqual(motion_parameters(asdict(skill.current)), skill.current)
        self.assertIsNone(motion_parameters(None))

    def test_hard_bounds_nonfinite_and_boolean_fail_before_motion(self):
        for name, (low, high) in ParameterizedKeeperMotion.BOUNDS.items():
            for value in (low-.001, high+.001, math.nan, math.inf, True, '1'):
                with self.subTest(name=name, value=value):
                    mapping = dict(asdict(MotionParameters()), **{name: value})
                    with self.assertRaises(ValueError):
                        motion_parameters(mapping)
            for value in (low, high):
                mapping = dict(asdict(MotionParameters()), **{name: value})
                self.assertEqual(getattr(motion_parameters(mapping), name), value)

    def test_exact_mapping_prevents_silent_missing_or_unknown_fields(self):
        with self.assertRaises(ValueError):
            motion_parameters({'guard_time_scale': 1.})
        with self.assertRaises(ValueError):
            motion_parameters(dict(asdict(MotionParameters()), root_velocity=1.))
        with self.assertRaises(TypeError):
            ParameterizedKeeperMotion.validate([1., 1., 0., 0.])


class KeeperMotionAdapterTests(unittest.TestCase):
    def sim(self, profile):
        sim = MotionPhysicalSimulator.__new__(MotionPhysicalSimulator)
        sim.motion_parameters = profile
        sim.profile_request_count = 0
        sim.motion_evidence = None
        sim.max_keeper_joint_velocity = 0.
        sim.outputs = {'keeper': {}}
        sim.adapter = Mock()
        sim.adapter.frame.return_value = {'velocity': [0., -2., 1.]}
        sim.bridges = {}
        return sim

    def native(self, sim, requests):
        profile = requests.get('keeper', {}).get('keeper_motion_skill')
        sim.outputs['keeper'] = ({} if profile is None else {
            'keeper_motion_skill': dict(profile, applied=True, changed_keyframes=2,
                                        changed_angles=4, clamped_angles=0, blocks=[{'id': 'genuflect'}])})
        return 'observations'

    def test_injection_only_changes_keeper_block_and_never_callers_mapping(self):
        sim = self.sim(MotionParameters(.9, 1.1, .02, .03))
        requests = {'keeper': {'motion': 'block', 'block_side': 'center'},
                    'striker': {'motion': 'kick'}}
        with patch('penalty.physical_sim.PhysicalSimulator.step', autospec=True,
                   side_effect=lambda target, req: self.native(target, req)) as native:
            self.assertEqual(sim.step(requests), 'observations')
        submitted = native.call_args.args[1]
        self.assertIn('keeper_motion_skill', submitted['keeper'])
        self.assertNotIn('keeper_motion_skill', submitted['striker'])
        self.assertNotIn('keeper_motion_skill', requests['keeper'])
        self.assertEqual(sim.profile_request_count, 1)
        self.assertEqual(sim.motion_evidence['blocks'][0]['id'], 'genuflect')
        self.assertEqual(sim.max_keeper_joint_velocity, 2.)

    def test_legacy_prepare_and_recovery_omit_extension(self):
        for profile, motion, side in ((None, 'block', 'center'),
                                      (MotionParameters(), 'block', 'prepare'),
                                      (MotionParameters(), 'getup', 'left'),
                                      (MotionParameters(), 'stand', 'center')):
            sim = self.sim(profile)
            with patch('penalty.physical_sim.PhysicalSimulator.step', autospec=True,
                       side_effect=lambda target, req: self.native(target, req)) as native:
                sim.step({'keeper': {'motion': motion, 'block_side': side}})
            self.assertNotIn('keeper_motion_skill', native.call_args.args[1]['keeper'])
            self.assertEqual(sim.profile_request_count, 0)

    def test_changed_confirmation_fails_and_float32_confirmation_matches(self):
        sim = self.sim(MotionParameters(.85, 1.15, .12, .1))
        def response(target, req):
            result = self.native(target, req)
            target.outputs['keeper']['keeper_motion_skill']['parameters']['guard_time_scale'] += 2e-8
            return result
        with patch('penalty.physical_sim.PhysicalSimulator.step', autospec=True, side_effect=response):
            sim.step({'keeper': {'motion': 'block', 'block_side': 'left'}})
        def wrong(target, req):
            result = self.native(target, req)
            target.outputs['keeper']['keeper_motion_skill']['parameters']['guard_time_scale'] = 1.
            return result
        with patch('penalty.physical_sim.PhysicalSimulator.step', autospec=True, side_effect=wrong):
            with self.assertRaises(RuntimeError):
                sim.step({'keeper': {'motion': 'block', 'block_side': 'left'}})

    def test_compact_native_updates_keep_first_applied_target_evidence(self):
        sim = self.sim(MotionParameters())
        with patch('penalty.physical_sim.PhysicalSimulator.step', autospec=True,
                   side_effect=lambda target, req: self.native(target, req)):
            sim.step({'keeper': {'motion': 'block', 'block_side': 'center'}})
        first = sim.motion_evidence
        def compact(target, req):
            result = self.native(target, req)
            del target.outputs['keeper']['keeper_motion_skill']['blocks']
            return result
        with patch('penalty.physical_sim.PhysicalSimulator.step', autospec=True, side_effect=compact):
            sim.step({'keeper': {'motion': 'block', 'block_side': 'center'}})
        self.assertEqual(sim.motion_evidence, first)

    def test_profile_change_waits_for_reset_and_native_reset_precedes_change(self):
        world = MotionDuelWorld.__new__(MotionDuelWorld)
        world.next_motion_parameters = MotionParameters()
        session = Mock()
        world._ensure_session = Mock(return_value=session)
        events = []
        session.sim.close.side_effect = lambda: events.append('close')
        session.sim.set_profile.side_effect = lambda _: events.append('profile')
        new_profile = MotionParameters(.9, 1., 0., .03)
        world.set_profile(new_profile)
        session.sim.set_profile.assert_not_called()
        with patch('penalty.rl.defense_benchmark.SensorDefenseDuelWorld.reset',
                   return_value='reset') as reset:
            self.assertEqual(world.reset('rng', {}), 'reset')
        self.assertEqual(events, ['close', 'profile'])
        session.sim.set_profile.assert_called_once_with(new_profile)
        reset.assert_called_once_with('rng', {})

    def test_active_simulator_cannot_change_profile(self):
        sim = self.sim(MotionParameters())
        sim.bridges = {'keeper': Mock()}
        with self.assertRaises(RuntimeError):
            sim.set_profile(None)
        self.assertEqual(sim.motion_parameters, MotionParameters())


if __name__ == '__main__':
    unittest.main()
