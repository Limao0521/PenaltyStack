"""Contract checks against the compiled original engine, never a fake runtime."""
import json
import pathlib
import subprocess
import unittest

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
    def test_native_hello(self):
        self.assertTrue(EXE.is_file(), 'Build original engine with tools/build_motion_runtime.py')
        process = subprocess.run([str(EXE)], input='{"type":"SHUTDOWN"}\n',
                                 text=True, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        hello = json.loads(process.stdout.splitlines()[0])
        self.assertEqual(hello['source_commit'], '2360ab1100ad28f70e190c893f6b095651ff0bbc')
        self.assertEqual(len(hello['joint_names']), 26)
        self.assertEqual(hello['cycle_s'], .012)

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

    def test_original_keeper_keyframes_and_mirror(self):
        for side, expected, mirror in (('center','genuflectStand',False),
                                      ('left','keeperJumpLeft',False),
                                      ('right','keeperJumpLeft',True)):
            with self.subTest(side=side):
                result, messages = run_native([{'motion':'stand'}] * 170 +
                    [{'motion':'block','block_side':side}] * 220)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-800:])
                outputs = [m for m in messages if m['type'] == 'OUTPUT']
                active = [m for m in outputs if m['phase'] == 'keyframeMotion']
                self.assertTrue(active)
                self.assertEqual(active[-1]['motion_info']['keyframe'], expected)
                self.assertEqual(active[-1]['motion_info']['mirror'], mirror)

    def test_oversized_or_truncated_messages_fail(self):
        for payload, code in (('[' * 65537 + '\n', 'MESSAGE_SIZE'),
                              ('{"type":"SHUTDOWN"}', 'TRUNCATED_MESSAGE')):
            result = subprocess.run([str(EXE)],input=payload,text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,2)
            self.assertEqual(json.loads(result.stdout.splitlines()[-1])['code'],code)
