"""Calibración offline de los umbrales naranjas de CopaOrangeBallPerceptor."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import tune_ball_thresholds as tune  # noqa: E402

CURRENT = tune.Thresholds(40, 235, 115, 150, 80)
SHAPE = {'gridStep': 4, 'minBlobCells': 3, 'minFillRatio': 0.45, 'maxAspectRatio': 2.0}
# YUV de referencia del comentario de copaOrangeBallPerceptor.cfg.
BALL = (135, 52, 213)
WOOD = (166, 96, 152)
SKIN = (170, 105, 156)
LINE = (30, 128, 128)
BLUE_POST = (90, 160, 110)
DISTRACTOR = (150, 100, 185)  # «casi naranja»: V-U=85 ≥ minChroma=80 → FP con los umbrales actuales


def make_yuv(width, height, base, disks=()):
    y = np.full((height, width), base[0], np.uint8)
    u = np.full((height, width), base[1], np.uint8)
    v = np.full((height, width), base[2], np.uint8)
    yy, xx = np.ogrid[:height, :width]
    for cx, cy, radius, color in disks:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
        y[mask], u[mask], v[mask] = color
    return y, u, v


def make_capture(name, kind, width, height, base, disks=(), distance_cm=None, step=4, clip=0):
    capture = tune.Capture(name, kind, make_yuv(width, height, base, disks), distance_cm)
    tune.sample(capture, step, clip)
    return capture


def write_yuv422(path, yuv):
    y, u, v = yuv
    height, width = y.shape
    blocks = np.empty((height, width // 2, 4), np.uint8)
    blocks[..., 0] = y[:, 0::2]
    blocks[..., 1] = u[:, 0::2]
    blocks[..., 2] = y[:, 1::2]
    blocks[..., 3] = v[:, 0::2]
    Path(path).write_bytes(blocks.tobytes())


class ColorConversionTest(unittest.TestCase):
    def test_rgb_to_yuv_matches_the_cfg_reference(self):
        y, u, v = tune.rgb_to_yuv(np.array([[[255, 100, 0]]], dtype=np.uint8))
        self.assertEqual((int(y[0, 0]), int(u[0, 0])), (135, 52))
        self.assertIn(int(v[0, 0]), (213, 214))  # el comentario del .cfg redondea a 213

    def test_yuv422_round_trip(self):
        y, u, v = make_yuv(16, 8, WOOD)
        # Rectángulo alineado a pares de columnas: U/V son uniformes dentro de cada par YUYV.
        y[2:6, 6:10], u[2:6, 6:10], v[2:6, 6:10] = BALL
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frame.yuv'
            write_yuv422(path, (y, u, v))
            loaded = tune.load_yuv422(path, 16, 8)
        for original, back in zip((y, u, v), loaded):
            self.assertTrue(np.array_equal(original, back))

    def test_yuv422_rejects_bad_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frame.yuv'
            path.write_bytes(b'\x00' * 10)
            with self.assertRaises(ValueError):
                tune.load_yuv422(path, 16, 8)
            with self.assertRaises(ValueError):
                tune.load_yuv422(path, 15, 8)


class SegmentationMirrorTest(unittest.TestCase):
    def test_ball_on_wood_is_detected_with_current_thresholds(self):
        capture = make_capture('ball', 'ball', 128, 96, WOOD, [(64, 48, 12, BALL)])
        count, blobs = tune.evaluate(capture, CURRENT, SHAPE, SHAPE['minBlobCells'])
        self.assertEqual(len(blobs), 1)
        self.assertGreater(count, 10)
        center, radius = blobs[0]['center'], blobs[0]['radius_px']
        self.assertAlmostEqual(center[0], 64, delta=4)
        self.assertAlmostEqual(center[1], 48, delta=4)
        self.assertAlmostEqual(radius, 12, delta=4)
        self.assertGreaterEqual(blobs[0]['fill'], SHAPE['minFillRatio'])

    def test_backgrounds_and_distractors_produce_no_blob(self):
        for base, disks, name in (
                (WOOD, [], 'duela vacía'),
                (WOOD, [(30, 30, 10, SKIN)], 'piel'),
                (WOOD, [(30, 30, 10, LINE), (90, 60, 8, BLUE_POST)], 'línea y poste'),
                (SKIN, [], 'todo piel')):
            with self.subTest(name):
                capture = make_capture('bg', 'bg', 128, 96, base, disks)
                self.assertEqual(tune.evaluate(capture, CURRENT, SHAPE, SHAPE['minBlobCells'])[1], [])

    def test_almost_orange_distractor_is_a_false_positive_with_current_thresholds(self):
        capture = make_capture('bg_distractores', 'bg', 128, 96, WOOD, [(64, 48, 10, DISTRACTOR)])
        count, blobs = tune.evaluate(capture, CURRENT, SHAPE, SHAPE['minBlobCells'])
        self.assertEqual(len(blobs), 1, 'V-U=85 >= minChroma=80: el distractor debe colarse')
        self.assertGreater(count, 5)

    def test_shape_filters_reject_sparse_blobs(self):
        # 4 celdas clasificadas aisladas, en puntos exactos de la rejilla (x=4gx+2):
        # ninguna alcanza minBlobCells=3 celdas conexas.
        capture = tune.Capture('sparse', 'bg', make_yuv(128, 96, WOOD))
        y, u, v = capture.y, capture.u, capture.v
        for cx, cy in ((18, 18), (58, 30), (98, 50), (38, 78)):
            y[cy, cx], u[cy, cx], v[cy, cx] = BALL
        tune.sample(capture, 4, 0)
        count, blobs = tune.evaluate(capture, CURRENT, SHAPE, SHAPE['minBlobCells'])
        self.assertEqual(count, 4)
        self.assertEqual(blobs, [])

    def test_clip_bottom_ignores_lower_rows(self):
        yuv = make_yuv(128, 96, WOOD, [(64, 88, 10, BALL)])
        clipped = tune.Capture('ball', 'ball', yuv)
        tune.sample(clipped, 4, clip_bottom=24)
        self.assertEqual(tune.evaluate(clipped, CURRENT, SHAPE, SHAPE['minBlobCells'])[1], [],
                         'el disco a y=88 cae en la franja recortada (y >= 96-24)')


class SuggestTest(unittest.TestCase):
    def setUp(self):
        self.balls = [make_capture('ball_%d' % radius, 'ball', 128, 96, WOOD,
                                   [(64, 48, radius, BALL)], distance_cm=distance)
                      for radius, distance in ((14, 20), (8, 130), (6, 200))]
        self.backgrounds = [
            make_capture('bg_cancha', 'bg', 128, 96, WOOD, [(30, 70, 8, LINE)]),
            make_capture('bg_piel', 'bg', 128, 96, SKIN),
            make_capture('bg_distractores', 'bg', 128, 96, WOOD, [(64, 48, 10, DISTRACTOR)]),
        ]

    def test_current_thresholds_are_not_usable_but_suggestion_is(self):
        rows = tune.evaluate_set(self.backgrounds + self.balls, CURRENT, SHAPE, 128)
        verdicts = {row['image']: row['verdict'] for row in rows}
        self.assertEqual(verdicts['bg_distractores'], 'FALSO_POSITIVO')
        for capture in self.balls:
            self.assertEqual(verdicts[capture.name], 'DETECTA')

        result = tune.suggest(self.balls, self.backgrounds, CURRENT, SHAPE, tune.QUICK)
        self.assertIsNotNone(result)
        thresholds = result['thresholds']
        rows = tune.evaluate_set(self.backgrounds + self.balls, thresholds, SHAPE, 128)
        for row in rows:
            self.assertIn(row['verdict'], ('SIN_BLOBS', 'DETECTA'), row['image'])
        self.assertGreater(result['bg_margin'], 0)
        self.assertGreaterEqual(result['ball_margin'], 0)
        self.assertEqual(result['score'], min(result['bg_margin'], result['ball_margin']))

    def test_suggestion_is_deterministic(self):
        first = tune.suggest(self.balls, self.backgrounds, CURRENT, SHAPE, tune.QUICK)
        second = tune.suggest(self.balls, self.backgrounds, CURRENT, SHAPE, tune.QUICK)
        self.assertEqual(first['thresholds'], second['thresholds'])
        self.assertEqual(first['score'], second['score'])

    def test_infeasible_set_returns_none(self):
        # Un «balón» del color de la madera: ningún umbral distingue las capturas.
        fake = [make_capture('ball_fake', 'ball', 128, 96, WOOD, [(64, 48, 10, WOOD)])]
        self.assertIsNone(tune.suggest(fake, self.backgrounds, CURRENT, SHAPE,
                                       {'maxU': (115,), 'minV': (150,), 'minChroma': (80,)}))


class MarginTest(unittest.TestCase):
    def test_bg_margin_is_the_relaxation_distance_to_the_closest_cell(self):
        capture = make_capture('bg', 'bg', 32, 32, WOOD)  # V-U = 152-96 = 56
        margin = tune.bg_margin([capture.sampled], CURRENT)
        self.assertEqual(margin, 80 - 56)  # minChroma es la restricción más cercana

    def test_bg_margin_zero_when_a_cell_already_classifies(self):
        capture = make_capture('bg', 'bg', 32, 32, DISTRACTOR)
        self.assertEqual(tune.bg_margin([capture.sampled], CURRENT), 0)

    def test_ball_margin_counts_tightening_until_detection_is_lost(self):
        capture = make_capture('ball', 'ball', 128, 96, WOOD, [(64, 48, 12, BALL)])
        margin = tune.ball_margin([capture], CURRENT, SHAPE)
        self.assertGreater(margin, 0)
        self.assertTrue(tune.evaluate(capture, CURRENT.tightened(margin), SHAPE,
                                      SHAPE['minBlobCells'])[1])
        self.assertFalse(tune.evaluate(capture, CURRENT.tightened(margin + 1), SHAPE,
                                       SHAPE['minBlobCells'])[1])


class CaptureLoadingTest(unittest.TestCase):
    def build_tree(self, root):
        for label, base, disks in (('ball_130', WOOD, [(32, 24, 10, BALL)]),
                                   ('bg_cancha', WOOD, [])):
            folder = Path(root) / label
            folder.mkdir(parents=True)
            write_yuv422(folder / ('%s_c0_000.yuv' % label), make_yuv(64, 48, base, disks))
            (folder / 'capture_meta.json').write_text(json.dumps(
                {'width': 64, 'height': 48, 'format': 'yuv422'}), encoding='utf-8')
        (Path(root) / 'notas.txt').write_text('ignorado', encoding='utf-8')
        (Path(root) / 'sueltas').mkdir()
        write_yuv422(Path(root) / 'sueltas' / 'foto.yuv', make_yuv(64, 48, WOOD))

    def test_load_captures_classifies_by_label_and_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build_tree(tmp)
            captures, skipped = tune.load_captures(tmp, tune.BALL_GLOBS, tune.BG_GLOBS)
        self.assertEqual([c.kind for c in captures], ['ball', 'bg'])
        self.assertEqual(captures[0].distance_cm, 130)
        self.assertIsNone(captures[1].distance_cm)
        self.assertEqual(skipped, ['sueltas/foto.yuv'])

    def test_yuv_without_dimensions_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'ball_020'
            folder.mkdir()
            write_yuv422(folder / 'ball_020_c0_000.yuv', make_yuv(64, 48, WOOD))
            with self.assertRaises(SystemExit):
                tune.load_captures(tmp, tune.BALL_GLOBS, tune.BG_GLOBS)


class EndToEndTest(unittest.TestCase):
    def test_main_reports_and_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            captures = Path(tmp) / 'captures'
            for label, base, disks in (('ball_020', WOOD, [(64, 48, 14, BALL)]),
                                       ('ball_130', WOOD, [(64, 48, 8, BALL)]),
                                       ('bg_cancha', WOOD, []),
                                       ('bg_distractores', WOOD, [(64, 48, 10, DISTRACTOR)])):
                folder = captures / label
                folder.mkdir(parents=True)
                write_yuv422(folder / ('%s_c0_000.yuv' % label), make_yuv(128, 96, base, disks))
                (folder / 'capture_meta.json').write_text(json.dumps(
                    {'width': 128, 'height': 96, 'format': 'yuv422'}), encoding='utf-8')
            cfg = Path(tmp) / 'copaOrangeBallPerceptor.cfg'
            shutil.copy(tune.DEFAULT_CFG, cfg)
            report = Path(tmp) / 'report.json'

            code = tune.main(['--captures', str(captures), '--cfg', str(cfg),
                              '--report', str(report), '--quick'])
            self.assertEqual(code, 0)
            data = json.loads(report.read_text(encoding='utf-8'))
            self.assertFalse(data['current']['usable'], 'el distractor debe ser FP con los actuales')
            self.assertIsNotNone(data['suggestion'])
            for row in data['suggestion']['per_image']:
                self.assertIn(row['verdict'], ('SIN_BLOBS', 'DETECTA'), row['image'])

            code = tune.main(['--captures', str(captures), '--cfg', str(cfg), '--quick', '--apply'])
            self.assertEqual(code, 0)
            backup = cfg.with_suffix('.cfg.bak')
            self.assertTrue(backup.is_file())
            values = tune.parse_cfg(cfg)
            suggested = data['suggestion']['thresholds']
            for key, value in suggested.items():
                self.assertEqual(values[key], str(value))
            untouched = tune.parse_cfg(backup)
            for key in ('gridStep', 'minBlobCells', 'minFillRatio', 'maxAspectRatio',
                        'minRadiusRatio', 'maxRadiusRatio', 'maxDistance', 'useUpperCamera'):
                self.assertEqual(values[key], untouched[key], key)
            self.assertIn('Tune with images from the venue', cfg.read_text(encoding='utf-8'),
                          'los comentarios del .cfg se conservan')


if __name__ == '__main__':
    unittest.main()
