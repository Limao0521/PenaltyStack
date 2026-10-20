"""Driver PC de las capturas de calibración del balón y script onboard (comandos y validaciones)."""
import os
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
# Este módulo se importa antes que test_package en el descubrimiento (orden alfabético):
# insertar ROOT/'deploy' haría que deploy/deploy.py sombreara el paquete `deploy` y
# fallaría `from deploy.package import package`. Se importa vía namespace desde ROOT.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import capture_ball_calibration as calib  # noqa: E402
from deploy import robot_deploy  # noqa: E402

LOKI = robot_deploy.Robot('loki', '192.168.49.3', '')


def options(**overrides):
    defaults = dict(label='ball_130', cameras='0,1', count=20, interval_ms=200, warmup=15,
                    resolution=2, format='auto', manage_service=True)
    defaults.update(overrides)
    return calib.CaptureOptions(**defaults)


class CommandTest(unittest.TestCase):
    def test_push_uploads_the_capture_script_over_the_build_ssh_setup(self):
        command = calib.push_command(LOKI)
        for part in ('source ./setupSSH', 'mkdir -p /home/nao/ball_calibration',
                     'rsync', 'nao@192.168.49.3', 'nao_capture_ball.py'):
            self.assertIn(part, command)

    def test_capture_stops_bhuman_runs_the_script_and_restarts(self):
        command = calib.capture_command(LOKI, options())
        for part in ('systemctl --user stop bhuman.service',
                     'python /home/nao/ball_calibration/nao_capture_ball.py',
                     '--label ball_130', '--cameras 0,1', '--count 20', '--interval-ms 200',
                     '--warmup 15', '--resolution 2', '--format auto',
                     'systemctl --user start bhuman.service', 'exit $rc'):
            self.assertIn(part, command)

    def test_keep_service_leaves_bhuman_alone(self):
        command = calib.capture_command(LOKI, options(manage_service=False))
        self.assertNotIn('systemctl', command)
        self.assertIn('--label ball_130', command)

    def test_fetch_uses_rsync_with_the_ssh_options(self):
        command = calib.fetch_command(LOKI, 'ball_130', ROOT / 'runs/ball_calibration/loki/ball_130')
        for part in ('rsync -rzce "ssh $SSHOPTIONS"',
                     'nao@192.168.49.3:/home/nao/ball_calibration/ball_130/'):
            self.assertIn(part, command)
        if os.name == 'nt':
            self.assertIn('/mnt/c/', command)

    def test_fetch_without_label_brings_everything(self):
        command = calib.fetch_command(LOKI, None, ROOT / 'runs/ball_calibration/loki')
        self.assertIn('nao@192.168.49.3:/home/nao/ball_calibration/', command)

    def test_local_destination_layout(self):
        self.assertEqual(calib.local_destination(LOKI, 'ball_130'),
                         ROOT / 'runs/ball_calibration/loki/ball_130')
        self.assertEqual(calib.local_destination(LOKI, None),
                         ROOT / 'runs/ball_calibration/loki')


class ValidationTest(unittest.TestCase):
    def test_valid_options_pass(self):
        calib.validate(options(label='bg_cancha', cameras='0', count=1, interval_ms=50,
                               warmup=0, resolution=1, format='yuv422'))

    def test_labels_and_cameras_are_shell_safe(self):
        for label in ('ball; rm -rf /', 'ball 130', 'balón', '$(reboot)', 'a' * 65, ''):
            with self.assertRaises(ValueError, msg=label):
                calib.validate(options(label=label))
        for cameras in ('2', '0,2', '0,1,1', '', 'a'):
            with self.assertRaises(ValueError, msg=cameras):
                calib.validate(options(cameras=cameras))

    def test_numeric_ranges(self):
        for bad in (dict(count=0), dict(count=201), dict(interval_ms=49), dict(interval_ms=5001),
                    dict(warmup=-1), dict(warmup=101), dict(resolution=9), dict(format='raw')):
            with self.assertRaises(ValueError, msg=bad):
                calib.validate(options(**bad))

    def test_fetch_rejects_bad_labels(self):
        with self.assertRaises(ValueError):
            calib.fetch_command(LOKI, 'x; reboot', ROOT)


class ImportHygieneTest(unittest.TestCase):
    def test_deploy_remains_a_namespace_package_after_importing_this_module(self):
        # Regresión: si capture_ball_calibration insertara ROOT/'deploy' en sys.path,
        # deploy/deploy.py sombrearía el paquete y test_package fallaría al importarlo.
        import deploy
        from deploy import package  # noqa: F401
        self.assertIsNone(getattr(deploy, '__file__', None),
                          'deploy debe seguir siendo un paquete de espacio de nombres')


class OnboardScriptTest(unittest.TestCase):
    """El script que corre en el robot no se puede ejecutar aquí; se valida su forma."""

    def setUp(self):
        self.source = (ROOT / 'tools/nao_capture_ball.py').read_text(encoding='utf-8')

    def test_is_valid_python_and_python2_compatible(self):
        compile(self.source, 'nao_capture_ball.py', 'exec')  # sintaxis py3
        self.assertNotRegex(self.source, re.compile(r'\bf["\']'), 'nada de f-strings (py2.7)')
        self.assertIn('from __future__ import print_function', self.source)
        self.assertNotIn('nonlocal ', self.source)

    def test_probes_native_yuv_before_rgb(self):
        order = [fmt for _, fmt, _ in re.findall(
            r'\((\d+), \'(\w+)\', (\d+)\)', self.source)]
        self.assertEqual(order[:2], ['yuv422', 'yuv422'])

    def test_never_guesses_the_buffer_position(self):
        self.assertIn('def find_buffer', self.source)
        self.assertNotIn('image[6]', self.source)


if __name__ == '__main__':
    unittest.main()
