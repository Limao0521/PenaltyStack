"""Actual articulated original-engine acceptance in the MuJoCo world."""
import json
import unittest
from pathlib import Path

import numpy as np

from penalty.motion_contract import ROOT
from penalty.physical_sim import PhysicalSimulator
from penalty.sim_session import SimulationSession


class PhysicalAcceptanceTests(unittest.TestCase):
    metrics = {}

    @classmethod
    def tearDownClass(cls):
        path = ROOT/'runs'/'acceptance.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(cls.metrics,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    def session(self, **kwargs):
        sim = PhysicalSimulator(**kwargs)
        self.addCleanup(sim.close)
        return SimulationSession(sim,record_dir=None,mode='manual')

    def settle(self, session):
        session.command('start')
        for _ in range(418):
            session.tick()
            if session.state == 'RUNNING':
                return
        self.fail('both robots failed to settle')

    def test_stand_ten_seconds_and_walk_five_seconds(self):
        session = self.session(ball_xy=(1.,.7))
        self.settle(session)
        for _ in range(834):
            session.tick()
            self.assertFalse(any(session.sim.fallen.values()))
        self.assertTrue(session.sim.stable())
        sim = session.sim
        start = sim.pose('striker')[0].copy()
        session.command('manual',{'motion':'walk','forward':.06})
        for _ in range(417):
            session.tick()
            self.assertFalse(any(sim.fallen.values()))
        displacement = float(np.linalg.norm(sim.pose('striker')[0][:2]-start[:2]))
        self.assertGreater(displacement,.1)
        self.assertEqual(sim.outputs['striker']['phase'],'walk')
        self.metrics['stand_walk'] = {'stand_s':10.008,'walk_s':5.004,
                                     'displacement_m':displacement,'fallen':sim.fallen}

    def test_left_kick_has_contact_and_both_original_actions_complete_once(self):
        cases = {}
        for kick in ('forwardFastLeft','forwardFastRight'):
            with self.subTest(kick=kick):
                session = self.session()
                session.command('manual',{'motion':'kick','kick_type':kick})
                self.settle(session)
                for _ in range(500):
                    session.tick()
                sim = session.sim
                displacement = float(np.linalg.norm(np.array(sim.snapshot()['ball_position'])[:2]-[.1,0]))
                self.assertEqual(sim.kick_count,1)
                if kick == 'forwardFastLeft':
                    self.assertGreater(sim.kick_contact_count,0)
                    self.assertGreater(displacement,.1)
                self.assertTrue(session._kick_done)
                self.assertFalse(sim.fallen['striker'])
                cases[kick] = {'ball_displacement_m':displacement,'kick_count':sim.kick_count,
                              'foot_ball_contacts':sim.kick_contact_count,'completed':session._kick_done,
                              'physical_contact_pass':sim.kick_contact_count>0 and displacement>.1}
                sim.close()
        self.metrics['kicks'] = cases

    def test_three_blocks_execute_original_keyframes(self):
        cases = {}
        for side in ('center','left','right'):
            with self.subTest(side=side):
                session = self.session(ball_xy=(1.,.8))
                session.command('role','keeper')
                session.command('manual',{'motion':'block','block_side':side})
                self.settle(session)
                sim = session.sim
                before = sim.data.qpos.copy()
                phases = set()
                max_change = 0.
                for _ in range(230):
                    session.tick()
                    phases.add(sim.outputs['keeper']['phase'])
                    max_change = max(max_change,float(np.max(abs(sim.data.qpos-before))))
                self.assertIn('keyframeMotion',phases)
                self.assertGreater(max_change,.1)
                self.assertTrue(np.isfinite(sim.data.qvel).all())
                cases[side] = {'phases':sorted(phases),'max_qpos_change':max_change,
                    'fallen':sim.fallen['keeper'],'position':sim.pose('keeper')[0].tolist()}
                sim.close()
        self.metrics['blocks'] = cases

    def test_keeper_sit_holds_guard_without_diving(self):
        session = self.session(ball_xy=(1.,.8))
        session.command('role','keeper')
        session.command('manual',{'motion':'sit'})
        self.settle(session)
        sim = session.sim
        standing_z = float(sim.pose('keeper')[0][2])
        goal_x = float(sim.pose('keeper')[0][0])
        phases, keyframes = set(), set()
        for _ in range(260):
            session.tick()
            output = sim.outputs['keeper']
            phases.add(output['phase'])
            if output['phase'] == 'keyframeMotion':
                keyframes.add(output['motion_info']['keyframe'])
        self.assertIn('keyframeMotion',phases)
        # Sentarse y mantener la guardia: el único keyframe es sitDownKeeper,
        # sin keeperJump* ni genuflectStand y sin caer al suelo.
        self.assertEqual(keyframes,{'sitDownKeeper'})
        final = sim.outputs['keeper']['motion_info']
        self.assertEqual(final['keyframe'],'sitDownKeeper')
        self.assertFalse(final['mirror'])
        self.assertFalse(sim.fallen['keeper'])
        seated_z = float(sim.pose('keeper')[0][2])
        self.assertLess(seated_z,standing_z-.08,'el torso no bajó: no está sentado')
        self.assertLess(abs(float(sim.pose('keeper')[0][0])-goal_x),.15,
                        'el portero se desplazó como en un lanzamiento')
        self.assertTrue(np.isfinite(sim.data.qvel).all())
        self.metrics['keeper_sit'] = {'standing_z_m':standing_z,'seated_z_m':seated_z,
            'keyframes':sorted(keyframes),'fallen':sim.fallen['keeper'],
            'position':sim.pose('keeper')[0].tolist()}
        sim.close()

    def test_three_repeats_deterministic_and_children_collected(self):
        session = self.session()
        session.mode = 'penalty'
        samples = []
        trials = []
        for repeat in range(3):
            session.command('repeat')
            trials.append(session.trial)
            keyframes = set()
            for _ in range(2800):
                session.tick()
                output = session.sim.outputs['keeper']
                if output.get('phase') == 'keyframeMotion':
                    keyframes.add(output['motion_info']['keyframe'])
                if session.state == 'FINISHED':
                    break
            self.assertEqual(session.state,'FINISHED')
            self.assertAlmostEqual(session.penalty_time,28.)
            self.assertTrue(all(c.state=='STOP' for c in session.controllers.values()))
            self.assertEqual(session.sim.kick_count,1)
            self.assertGreater(session.sim.kick_contact_count,0)
            self.assertGreater(session.sim.block_contacts,0)
            self.assertFalse(any(session.sim.fallen.values()))
            # El portero del ensayo sólo se sienta: nunca se lanza.
            self.assertEqual(keyframes,{'sitDownKeeper'})
            self.assertEqual(session.sim.outputs['keeper']['motion'],'sit')
            samples.append(session.sim.data.qpos.copy())
        self.assertEqual(len(set(trials)),3)
        for sample in samples[1:]:
            np.testing.assert_allclose(sample,samples[0],rtol=0,atol=1e-7)
        processes = [b.process for b in session.sim.bridges.values()]
        session.close()
        session.tick()
        self.assertTrue(all(p.poll() is not None for p in processes))
        self.metrics['repeats'] = {'count':3,'max_qpos_error':float(max(
            np.max(abs(sample-samples[0])) for sample in samples)),
            'result':session.sim.result(),'final':session.sim.snapshot(),'children_collected':True}


if __name__ == '__main__':
    unittest.main()
