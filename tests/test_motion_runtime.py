"""Contract checks against the compiled original engine, never a fake runtime."""
import json
import pathlib
import subprocess
import unittest
import math

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ROOT / 'native/bhuman_motion/build/Release/bhuman_motion.exe'


def frame(sequence, request):
    return {'type': 'STEP', 'trial': 'native-test', 'sequence': sequence,
            'time_ms': 10000 + 12 * sequence,
            'sensors': {'angles': [0.] * 26, 'velocity': [0.] * 26,
                        'gyro': [0.] * 3, 'acc': [0., 0., 9.81],
                        'orientation': [1., 0., 0., 0.], 'fsr': [.65] * 8},
            'request': request}


def run_native(requests):
    messages = [{'type': 'INIT', 'protocol': 1, 'trial': 'native-test',
                 'role': 'striker', 'seed': 7}]
    messages += [frame(i, request) for i, request in enumerate(requests)]
    messages.append({'type': 'SHUTDOWN'})
    result = subprocess.run([str(EXE)], input=''.join(json.dumps(m) + '\n' for m in messages),
                            text=True, capture_output=True, timeout=30)
    return result, [json.loads(line) for line in result.stdout.splitlines()]


class NativeMotionRuntimeTests(unittest.TestCase):
    def test_tilt_uses_original_angle_axis_without_yaw(self):
        # Original InertialDataProvider uses the horizontal rotation vector,
        # not Euler roll/pitch. They differ substantially when lying sideways.
        tilt = math.radians(120)
        axis = 2**-.5
        sensors = frame(0,{'motion':'stand'})
        sensors['sensors']['orientation'] = [math.cos(tilt/2),
            axis*math.sin(tilt/2),axis*math.sin(tilt/2),0.]
        messages = [{'type':'INIT','protocol':1,'trial':'native-test','role':'keeper','seed':0},
                    sensors,{'type':'SHUTDOWN'}]
        result = subprocess.run([str(EXE)],input=''.join(json.dumps(m)+'\n' for m in messages),
                                text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        output = json.loads(result.stdout.splitlines()[2])
        angles = output['sensor_info']['inertial_angle']
        self.assertAlmostEqual(angles[0],tilt*axis,places=5)
        self.assertAlmostEqual(angles[1],tilt*axis,places=5)
        self.assertAlmostEqual(angles[2],0.,places=5)

    def test_native_hello(self):
        self.assertTrue(EXE.is_file(), 'Build original engine with tools/build_motion_runtime.py')
        process = subprocess.run([str(EXE)], input='{"type":"SHUTDOWN"}\n',
                                 text=True, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        hello = json.loads(process.stdout.splitlines()[0])
        self.assertEqual(hello['source_commit'], '2360ab1100ad28f70e190c893f6b095651ff0bbc')
        self.assertEqual(len(hello['joint_names']), 26)
        self.assertEqual(hello['cycle_s'], .012)
        self.assertEqual(hello['protocol'], 1)
        self.assertIn('keeper_defenses_v2', hello['capabilities'])
        self.assertEqual(hello['keeper_defenses'],
                         ['center', 'center_right', 'wide_left', 'wide_right', 'left', 'right'])
        self.assertEqual(hello['keeper_preparations'], ['prepare'])

    def test_original_stand_and_walk_generate_joint_commands(self):
        # Constant ideal sensors test the native motion providers, not physical
        # stability. The latter is checked by the MuJoCo integration tests.
        result, messages = run_native([{'motion': 'stand'}] * 150 +
                                      [{'motion': 'walk', 'forward': 60., 'lateral': 0., 'turn': 0.}] * 180)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-800:])
        outputs = [m for m in messages if m['type'] == 'OUTPUT']
        self.assertEqual(len(outputs), 330)
        self.assertIn('walk', {m['phase'] for m in outputs[150:]})
        self.assertGreater(max(m['angles'][17] for m in outputs) - min(m['angles'][17] for m in outputs), .1)
        for output in outputs:
            self.assertEqual(len(output['angles']), 26)
            self.assertTrue(all(0 <= stiffness <= 100 for stiffness in output['stiffness']))

    def test_invalid_sequence_is_rejected(self):
        init = {'type': 'INIT', 'protocol': 1, 'trial': 'native-test', 'role': 'striker', 'seed': 1}
        result = subprocess.run([str(EXE)], input=json.dumps(init) + '\n' + json.dumps(frame(4, {'motion':'stand'})) + '\n',
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])['type'], 'ERROR')

    def test_original_kick_reports_phase_and_completion(self):
        result, messages = run_native([{'motion':'stand'}] * 170 +
            [{'motion':'kick','kick_type':'forwardFastLeft','ball':[160.,50.]}] * 350)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-800:])
        outputs = [m for m in messages if m['type'] == 'OUTPUT']
        self.assertIn('kick', {m['phase'] for m in outputs})
        self.assertTrue(any(m['kick_completed'] for m in outputs))

    def test_requested_kick_length_changes_original_trajectory(self):
        trajectories=[]
        for length in (1500.,2500.):
            result,messages = run_native([{'motion':'stand'}]*170 +
                [{'motion':'kick','kick_type':'forwardFastRight','ball':[160.,-50.],
                  'kick_length_mm':length}]*350)
            self.assertEqual(result.returncode,0,result.stderr)
            trajectories.append([m['angles'][23] for m in messages
                if m['type']=='OUTPUT' and m['phase']=='kick'])
        difference = max(abs(a-b) for a,b in zip(*trajectories))
        self.assertGreater(difference,.05)

    def test_original_keeper_keyframes_and_mirror(self):
        for side, expected, mirror in (('center','genuflectStand',False),
                                      ('center_right','genuflectStand',True),
                                      ('wide_left','genuflectStandDefender',False),
                                      ('wide_right','genuflectStandDefender',True),
                                      ('left','keeperJumpLeft',False),
                                      ('right','keeperJumpLeft',True),
                                      ('prepare','sitDownKeeper',False)):
            with self.subTest(side=side):
                result, messages = run_native([{'motion':'stand'}] * 170 +
                    [{'motion':'block','block_side':side}] * 220)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-800:])
                outputs = [m for m in messages if m['type'] == 'OUTPUT']
                active = [m for m in outputs if m['phase'] == 'keyframeMotion']
                self.assertTrue(active)
                self.assertEqual(active[-1]['motion_info']['keyframe'], expected)
                self.assertEqual(active[-1]['motion_info']['mirror'], mirror)

    def test_unknown_keeper_block_request_is_rejected(self):
        result, messages = run_native([{'motion':'block','block_side':'unknown_guard'}])
        self.assertEqual(result.returncode, 2)
        self.assertEqual(messages[-1]['type'], 'ERROR')
        self.assertEqual(messages[-1]['code'], 'INVALID_FRAME')
        self.assertIn('Unknown block side', messages[-1]['message'])

    def test_oversized_or_truncated_messages_fail(self):
        for payload, code in (('[' * 65537 + '\n', 'MESSAGE_SIZE'),
                              ('{"type":"SHUTDOWN"}', 'TRUNCATED_MESSAGE')):
            result = subprocess.run([str(EXE)],input=payload,text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,2)
            self.assertEqual(json.loads(result.stdout.splitlines()[-1])['code'],code)
