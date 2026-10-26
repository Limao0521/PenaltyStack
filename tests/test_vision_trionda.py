"""Visión simulada del balón Trionda: cámaras de cabeza + detector naranja.

Renderiza QVGA offscreen con las cámaras CameraBottom adjuntadas al cuerpo
Head y comprueba penalty.vision.detect_orange_ball contra el suelo verdadero
de la física. Tolerancias de baseline sin calibrar, como tools/vision_check.py.
Se omite si falta mujoco o un backend GL offscreen (EGL/OSMesa/display).
"""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

try:
    import mujoco
    _spec = importlib.util.spec_from_file_location(
        'vision_check', ROOT / 'tools' / 'vision_check.py')
    vision_check = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(vision_check)
except ImportError:  # sin mujoco ni siquiera se puede cargar la herramienta
    mujoco = None
    vision_check = None


@unittest.skipIf(mujoco is None or vision_check is None,
                 'mujoco no está instalado')
class TriondaVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from penalty.physical_sim import PhysicalSimulator
        cls.sim = PhysicalSimulator(seed=0)
        try:
            cls.renderer, cls.backend = vision_check.create_renderer(cls.sim.model)
        except RuntimeError as error:
            cls.sim.close()
            raise unittest.SkipTest(str(error))

    @classmethod
    def tearDownClass(cls):
        cls.renderer.close()
        cls.sim.close()

    def look(self, role, ball, yaw, pitch):
        self.sim.reset_world()
        vision_check.place_ball(self.sim, ball)
        vision_check.set_head(self.sim, role, yaw, pitch)
        return vision_check.evaluate(self.sim, self.renderer, role, None, '')

    def test_model_exposes_head_cameras_with_original_fovy(self):
        names = [self.sim.model.camera(index).name
                 for index in range(self.sim.model.ncam)]
        for role in ('striker', 'keeper'):
            for camera in ('CameraBottom', 'CameraTop'):
                self.assertIn(role + '_' + camera, names)
        from penalty.nao_model import CAMERA_FOVY_DEG
        camera = self.sim.model.camera('keeper_CameraBottom')
        self.assertAlmostEqual(float(camera.fovy[0]), CAMERA_FOVY_DEG, places=3)

    def test_trionda_material_is_bound_to_the_ball(self):
        geom = self.sim.model.geom('orange_ball')
        self.assertGreaterEqual(int(geom.matid[0]), 0,
                                'el balón debe usar el material trionda')
        self.assertEqual(self.sim.model.material(int(geom.matid[0])).name, 'trionda')

    def test_keeper_detects_trionda_ball_at_penalty_mark(self):
        row = self.look('keeper', (0.10, 0.0), 0.0, 0.0)
        self.assertTrue(row['truth']['in_frame'])
        self.assertTrue(row['detected'], 'el portero no ve el balón Trionda a 1.18 m')
        self.assertLess(row['range_error_m'],
                        max(vision_check.RANGE_ATOL,
                            vision_check.RANGE_RTOL * row['truth']['depth_m']))
        self.assertLess(row['bearing_error_rad'], vision_check.BEARING_TOL)

    def test_striker_detects_trionda_ball_at_its_feet_with_head_down(self):
        row = self.look('striker', (0.10, 0.0), 0.0, 0.45)
        self.assertTrue(row['truth']['in_frame'])
        self.assertTrue(row['detected'], 'el tirador no ve el balón a 0.20 m')
        self.assertLess(row['range_error_m'],
                        max(vision_check.RANGE_ATOL,
                            vision_check.RANGE_RTOL * row['truth']['depth_m']))
        self.assertLess(row['bearing_error_rad'], vision_check.BEARING_TOL)

    def test_striker_bottom_camera_is_blind_at_its_feet_without_head_pitch(self):
        # Limitación geométrica honesta: con head_pitch 0 el suelo visible de
        # CameraBottom empieza ~0.26 m por delante; el balón del punto penal
        # (0.20 m) queda fuera del cuadro hasta cabecear hacia abajo.
        row = self.look('striker', (0.10, 0.0), 0.0, 0.0)
        self.assertFalse(row['truth']['in_frame'])
        self.assertFalse(row['detected'])

    def test_no_detection_for_ball_behind_the_robot(self):
        row = self.look('striker', (0.90, 0.0), 0.0, 0.0)
        self.assertFalse(row['truth']['in_frame'])
        self.assertFalse(row['truth']['intersects'])
        self.assertFalse(row['detected'])


if __name__ == '__main__':
    unittest.main()
