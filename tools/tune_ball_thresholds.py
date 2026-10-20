"""Offline tuner for the orange-ball color thresholds of CopaOrangeBallPerceptor.

Mirrors the segmentation of robot/Src/Modules/Perception/BallPerceptors/CopaOrangeBallPerceptor.cpp
(YUV classification, grid sampling, 4-connected flood fill, blob shape filters) and evaluates
venue captures: ball shots must produce a valid blob, background shots must produce none. It is
the offline companion of the manual rules in docs/GUIA_DESPLIEGUE_ROBOT_FISICO.md §8; the full
workflow lives in docs/CALIBRACION_BALON.md.

Not mirrored (they need the robot at runtime): BodyContour clipping (--clip-bottom approximates
it), ground projection, maxDistance and the expected-radius gate (reported as a sanity check
when the label encodes the distance, e.g. ball_130 = 130 cm). The blob score here is
cells*fill without the size-mismatch factor.

    python tools/tune_ball_thresholds.py --captures runs/ball_calibration/falcanao
    python tools/tune_ball_thresholds.py --captures ... --report runs/ball_calibration/falcanao/report.json
    python tools/tune_ball_thresholds.py --captures ... --apply   # reescribe el .cfg (con respaldo)

Captures come from tools/capture_ball_calibration.py (native YUV422 .yuv + capture_meta.json)
or any .ppm/.png/.jpg (converted with BT.601, ±1 level vs the camera's own YUV).
"""
import argparse
import fnmatch
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / 'robot/Config/Scenarios/CopaNAOPenalty/copaOrangeBallPerceptor.cfg'
COLOR_KEYS = ('minY', 'maxY', 'maxU', 'minV', 'minChroma')
IMAGE_SUFFIXES = ('.yuv', '.ppm', '.png', '.jpg', '.jpeg', '.bmp')
BALL_GLOBS = ('ball*',)
BG_GLOBS = ('bg*',)
# Búsqueda por omisión: los tres umbrales que dominan madera clara, piel y reflejos.
SEARCH = {'maxU': tuple(range(85, 136, 5)), 'minV': tuple(range(120, 211, 5)),
          'minChroma': tuple(range(40, 151, 10))}
SEARCH_Y = {'minY': (25, 40, 60), 'maxY': (225, 235, 250)}
QUICK = {'maxU': tuple(range(85, 136, 10)), 'minV': tuple(range(120, 211, 10)),
         'minChroma': tuple(range(40, 151, 20))}
MARGIN_CAP = 999


class Thresholds(tuple):
    """(minY, maxY, maxU, minV, minChroma) — los cinco enteros del .cfg."""

    def __new__(cls, minY, maxY, maxU, minV, minChroma):
        return super().__new__(cls, (int(minY), int(maxY), int(maxU), int(minV), int(minChroma)))

    def tightened(self, delta):
        return Thresholds(self[0] + delta, self[1] - delta, self[2] - delta,
                          self[3] + delta, self[4] + delta)

    def as_dict(self):
        return dict(zip(COLOR_KEYS, self))


class Capture:
    """Una imagen etiquetada: kind='ball' exige detección, kind='bg' exige cero blobs."""

    def __init__(self, name, kind, yuv, distance_cm=None):
        self.name = name
        self.kind = kind
        self.y, self.u, self.v = yuv  # uint8, (height, width)
        self.distance_cm = distance_cm
        self.sampled = None  # (y, u, v) int16 planos de las celdas de la rejilla


def parse_cfg(path):
    """Claves del .cfg como cadenas ('gridStep = 4;' → {'gridStep': '4'})."""
    values = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        match = re.match(r'^\s*(\w+)\s*=\s*([^;]+);', line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def rgb_to_yuv(rgb):
    """BT.601 full-range, como el comentario del .cfg: RGB(255,100,0) → ~(135, 52, 213).

    El redondeo da (135, 52, 214); el comentario del .cfg usa 213 (±1 nivel, irrelevante
    frente a márgenes de decenas). Solo se usa para capturas no nativas (.ppm/.png/.jpg):
    el robot entrega YUV422 directo y esos frames no pasan por aquí.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return tuple(np.clip(np.rint(a), 0, 255).astype(np.uint8) for a in (y, u, v))


def load_yuv422(path, width, height):
    """Frame YUYV crudo (2 bytes/píxel: Y0 U Y1 V), el formato nativo de la cámara."""
    if width % 2:
        raise ValueError('ancho impar en %s' % path)
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if raw.size != width * height * 2:
        raise ValueError('%s: %d bytes, esperados %d para %dx%d YUV422'
                         % (path, raw.size, width * height * 2, width, height))
    blocks = raw.reshape(height, width // 2, 4)
    y = np.empty((height, width), dtype=np.uint8)
    y[:, 0::2] = blocks[..., 0]
    y[:, 1::2] = blocks[..., 2]
    u = np.repeat(blocks[..., 1], 2, axis=1)
    v = np.repeat(blocks[..., 3], 2, axis=1)
    return y, u, v


def load_rgb(path):
    from PIL import Image
    img = Image.open(path)
    if img.width % 2:  # la rejilla y YUV422 asumen ancho par
        img = img.crop((0, 0, img.width - 1, img.height))
    return rgb_to_yuv(np.asarray(img.convert('RGB'), dtype=np.uint8))


def sample(capture, step, clip_bottom):
    """Muestrea la rejilla exactamente como el perceptor: x=gx*step+step/2, y=gy*step+step/2.

    clip_bottom aproxima BodyContour: descarta filas a menos de clip_bottom píxeles del borde
    inferior (en el robot las recorta el contorno del propio cuerpo).
    """
    height, width = capture.y.shape
    grid_w, grid_h = width // step, height // step
    xs = np.arange(grid_w) * step + step // 2
    ys = np.arange(grid_h) * step + step // 2
    if clip_bottom > 0:
        ys = ys[ys < height - clip_bottom]
    grid = np.ix_(ys, xs)
    capture.sampled = (capture.y[grid].astype(np.int16).reshape(-1),
                       capture.u[grid].astype(np.int16).reshape(-1),
                       capture.v[grid].astype(np.int16).reshape(-1),
                       grid_w, grid_h, len(ys))


def classify(sampled, thresholds):
    """isOrange del perceptor sobre las celdas muestreadas: máscara plana de rows*grid_w."""
    y, u, v, grid_w, _grid_h, rows = sampled
    minY, maxY, maxU, minV, minChroma = thresholds
    mask = ((y >= minY) & (y <= maxY) & (u <= maxU) & (v >= minV) & (v - u >= minChroma))
    if rows == _grid_h:
        return mask
    # Celdas recortadas por BodyContour: nunca clasifican; la máscara vive en la rejilla completa.
    full = np.zeros(grid_w * _grid_h, dtype=bool)
    kept = (np.arange(rows)[:, None] * grid_w + np.arange(grid_w)[None, :]).reshape(-1)
    full[kept] = mask
    return full


def find_blobs(mask, grid_w, grid_h, step, min_fill_ratio, max_aspect_ratio):
    """Flood fill 4-conexo idéntico al de CopaOrangeBallPerceptor.cpp; devuelve todos los blobs."""
    pending = {int(i) for i in np.flatnonzero(mask)}
    blobs = []
    while pending:
        start = min(pending)
        pending.discard(start)
        stack = [start]
        component = []
        while stack:
            cell = stack.pop()
            cx, cy = cell % grid_w, cell // grid_w
            component.append((cx, cy))
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if 0 <= nx < grid_w and 0 <= ny < grid_h:
                    index = ny * grid_w + nx
                    if index in pending:
                        pending.discard(index)
                        stack.append(index)
        cells = len(component)
        xs = [c[0] for c in component]
        ys = [c[1] for c in component]
        box_w, box_h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        fill = cells / float(box_w * box_h)
        aspect = float(max(box_w, box_h)) / float(min(box_w, box_h))
        blobs.append({
            'cells': cells, 'fill': fill, 'aspect': aspect,
            'valid': bool(fill >= min_fill_ratio and aspect <= max_aspect_ratio),
            'center': ((min(xs) + max(xs) + 1) * step / 2.0, (min(ys) + max(ys) + 1) * step / 2.0),
            'radius_px': max(box_w, box_h) * step / 2.0,
            'score': cells * fill,
        })
    blobs.sort(key=lambda blob: -blob['score'])
    return blobs


def evaluate(capture, thresholds, shape, min_blob_cells):
    """(celdas clasificadas, blobs válidos) de una imagen con unos umbrales dados."""
    mask = classify(capture.sampled, thresholds)
    count = int(mask.sum())
    if count < min_blob_cells:
        return count, []
    grid_w, grid_h = capture.sampled[3], capture.sampled[4]
    blobs = find_blobs(mask, grid_w, grid_h, shape['gridStep'],
                       shape['minFillRatio'], shape['maxAspectRatio'])
    return count, [blob for blob in blobs if blob['valid'] and blob['cells'] >= min_blob_cells]


def bg_margin(sampled_list, thresholds):
    """Niveles YUV que se pueden relajar antes de que alguna celda de fondo empiece a clasificar.

    Conservador por construcción: ignora que además haría falta un blob válido (≥3 celdas
    conexas con fill/aspect aceptables). 0 significa que ya hay celdas clasificadas.
    """
    minY, maxY, maxU, minV, minChroma = thresholds
    best = MARGIN_CAP
    for y, u, v, _gw, _gh, _rows in sampled_list:
        if not y.size:
            continue
        viol = np.stack([minY - y, y - maxY, u - maxU, minV - v, minChroma - (v - u)])
        positive = viol > 0
        passing = ~positive.any(axis=0)
        dist = np.where(passing, 0, np.where(positive, viol, MARGIN_CAP * 4).min(axis=0))
        best = min(best, int(dist.min()))
    return best


def ball_margin(balls, thresholds, shape, cap=64):
    """Niveles que se pueden endurecer antes de perder el balón en alguna captura (bisect)."""
    def detects_all(candidate):
        for capture in balls:
            if not evaluate(capture, candidate, shape, shape['minBlobCells'])[1]:
                return False
        return True
    if not detects_all(thresholds):
        return -1
    low, high = 0, cap
    while low < high:
        mid = (low + high + 1) // 2
        if detects_all(thresholds.tightened(mid)):
            low = mid
        else:
            high = mid - 1
    return low


def suggest(balls, backgrounds, current, shape, search, on_progress=None):
    """Busca los umbrales factibles con mayor margen bilateral min(fondo, balón).

    Factible: cero blobs válidos en TODOS los fondos y ≥1 blob válido en TODAS las de balón.
    Desempate: menor distancia L1 a los umbrales actuales (cambiar lo mínimo).
    """
    ranges = dict(search)
    best = None  # (score, -l1, thresholds)
    total = 1
    for values in ranges.values():
        total *= len(values)
    done = 0
    for minY in ranges.get('minY', (current[0],)):
        for maxY in ranges.get('maxY', (current[1],)):
            for maxU in ranges['maxU']:
                for minV in ranges['minV']:
                    for minChroma in ranges['minChroma']:
                        done += 1
                        if on_progress and done % 256 == 0:
                            on_progress(done, total)
                        thresholds = Thresholds(minY, maxY, maxU, minV, minChroma)
                        # Fondos: el recuento vectorizado dentro de evaluate evita el flood
                        # fill cuando hay menos de minBlobCells celdas clasificadas.
                        if any(evaluate(c, thresholds, shape, shape['minBlobCells'])[1]
                               for c in backgrounds):
                            continue
                        if not all(evaluate(c, thresholds, shape, shape['minBlobCells'])[1]
                                   for c in balls):
                            continue
                        margin_bg = bg_margin([c.sampled for c in backgrounds], thresholds)
                        if best is not None and margin_bg < best[0]:
                            continue  # score = min(margin_bg, margin_ball) no puede superar al mejor
                        margin_ball = ball_margin(balls, thresholds, shape)
                        if margin_ball < 0:
                            continue
                        score = min(margin_bg, margin_ball)
                        l1 = sum(abs(a - b) for a, b in zip(thresholds, current))
                        key = (score, -l1, tuple(-t for t in thresholds))
                        if best is None or key > (best[0], best[1], best[2]):
                            best = (score, -l1, tuple(-t for t in thresholds), thresholds,
                                    margin_bg, margin_ball)
    if best is None:
        return None
    return {'thresholds': best[3], 'bg_margin': best[4], 'ball_margin': best[5],
            'score': min(best[4], best[5]), 'candidates': done}


def load_captures(root, ball_globs, bg_globs, width=None, height=None):
    """Recorre root/ y clasifica por nombre: ball* = balón, bg* = fondo (ver capture_meta.json)."""
    root = Path(root)
    captures, skipped = [], []
    metas = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        kind = 'ball' if any(fnmatch.fnmatch(rel, glob) for glob in ball_globs) else \
               'bg' if any(fnmatch.fnmatch(rel, glob) for glob in bg_globs) else None
        if kind is None:
            skipped.append(rel)
            continue
        meta_path = path.parent / 'capture_meta.json'
        if meta_path not in metas:
            metas[meta_path] = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
        meta = metas[meta_path]
        suffix = path.suffix.lower()
        if suffix == '.yuv':
            frame_w = width or meta.get('width')
            frame_h = height or meta.get('height')
            if not frame_w or not frame_h:
                raise SystemExit('%s: .yuv sin dimensiones; faltan capture_meta.json o --width/--height' % rel)
            yuv = load_yuv422(path, int(frame_w), int(frame_h))
        else:
            yuv = load_rgb(path)
        distance = re.search(r'ball_?0*(\d+)', path.parent.name)
        captures.append(Capture(rel, kind, yuv, int(distance.group(1)) if kind == 'ball' and distance else None))
    return captures, skipped


def radius_check(capture, blob, width, hfov=61.0, ball_radius_mm=32.5):
    """Compara el radio del blob con el esperado por pinhole a la distancia de la etiqueta.

    Aproximación del gate runtime minRadiusRatio/maxRadiusRatio (que usa la CameraMatrix real):
    solo avisa en el reporte, no participa en la búsqueda.
    """
    if capture.distance_cm is None or not blob:
        return None
    focal_px = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    expected = focal_px * ball_radius_mm / (capture.distance_cm * 10.0)
    if expected <= 0:
        return None
    ratio = blob[0]['radius_px'] / expected
    return {'expected_radius_px': round(expected, 1), 'ratio': round(ratio, 2),
            'ok': bool(0.5 <= ratio <= 2.0)}


def evaluate_set(captures, thresholds, shape, width, hfov=61.0):
    """Tabla de resultados por imagen (para pantalla y reporte JSON)."""
    rows = []
    for capture in captures:
        count, valid = evaluate(capture, thresholds, shape, shape['minBlobCells'])
        row = {'image': capture.name, 'kind': capture.kind, 'classified_cells': count,
               'valid_blobs': len(valid)}
        if valid:
            best = valid[0]
            row['best'] = {'cells': best['cells'], 'fill': round(best['fill'], 3),
                           'aspect': round(best['aspect'], 3),
                           'center': [round(c, 1) for c in best['center']],
                           'radius_px': best['radius_px']}
            check = radius_check(capture, valid, width, hfov)
            if check:
                row['radius_check'] = check
        if capture.kind == 'bg':
            row['verdict'] = 'SIN_BLOBS' if not valid else 'FALSO_POSITIVO'
        else:
            row['verdict'] = 'DETECTA' if valid else 'NO_DETECTA'
        rows.append(row)
    return rows


def print_table(rows):
    for row in rows:
        detail = ''
        if 'best' in row:
            best = row['best']
            detail = ' celdas=%d fill=%.2f aspect=%.2f centro=(%.0f,%.0f) r=%.1fpx' % (
                best['cells'], best['fill'], best['aspect'], best['center'][0],
                best['center'][1], best['radius_px'])
            if 'radius_check' in row:
                check = row['radius_check']
                detail += ' radio_esperado=%.1f ratio=%.2f%s' % (
                    check['expected_radius_px'], check['ratio'], '' if check['ok'] else ' ¡FUERA DE RANGO!')
        mark = {'DETECTA': 'ok ', 'SIN_BLOBS': 'ok ', 'NO_DETECTA': 'MAL', 'FALSO_POSITIVO': 'MAL'}[row['verdict']]
        print('  [%s] %-52s %-13s celdas=%-5d blobs=%d%s'
              % (mark, row['image'], row['verdict'], row['classified_cells'], row['valid_blobs'], detail))


def apply_thresholds(cfg_path, thresholds):
    """Reescribe solo las cinco claves de color del .cfg, conservando comentarios y demás."""
    cfg_path = Path(cfg_path)
    backup = cfg_path.with_suffix(cfg_path.suffix + '.bak')
    text = cfg_path.read_text(encoding='utf-8')
    backup.write_text(text, encoding='utf-8')
    for key, value in thresholds.as_dict().items():
        text, count = re.subn(r'(?m)^(%s\s*=\s*)[^;]+(;)' % key, r'\g<1>%d\g<2>' % value, text)
        if count != 1:
            raise SystemExit('no se pudo reescribir %s en %s (%d coincidencias)' % (key, cfg_path, count))
    cfg_path.write_text(text, encoding='utf-8')
    return backup


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Calibra los umbrales naranjas de CopaOrangeBallPerceptor con capturas de la sede.')
    parser.add_argument('--captures', required=True, help='carpeta con capturas (ball*/bg*)')
    parser.add_argument('--cfg', default=str(DEFAULT_CFG), help='copaOrangeBallPerceptor.cfg')
    parser.add_argument('--ball-globs', nargs='+', default=list(BALL_GLOBS))
    parser.add_argument('--bg-globs', nargs='+', default=list(BG_GLOBS))
    parser.add_argument('--clip-bottom', type=int, default=0,
                        help='píxeles inferiores a ignorar (aproxima BodyContour; 0 = nada)')
    parser.add_argument('--width', type=int, default=None, help='ancho de .yuv sin capture_meta.json')
    parser.add_argument('--height', type=int, default=None, help='alto de .yuv sin capture_meta.json')
    parser.add_argument('--hfov', type=float, default=61.0, help='FOV horizontal para el chequeo de radio')
    parser.add_argument('--search-y', action='store_true', help='buscar también minY/maxY (más lento)')
    parser.add_argument('--quick', action='store_true', help='rejilla de búsqueda más gruesa')
    parser.add_argument('--report', default=None, help='escribir reporte JSON en esta ruta')
    parser.add_argument('--apply', action='store_true', help='reescribir el .cfg con la sugerencia (con .bak)')
    args = parser.parse_args(argv)

    cfg = parse_cfg(args.cfg)
    current = Thresholds(*(cfg[key] for key in COLOR_KEYS))
    shape = {'gridStep': max(1, int(cfg.get('gridStep', 4))),
             'minBlobCells': int(cfg.get('minBlobCells', 3)),
             'minFillRatio': float(cfg.get('minFillRatio', 0.45)),
             'maxAspectRatio': float(cfg.get('maxAspectRatio', 2.0))}
    print('Umbrales actuales: %s · rejilla %d · blobs >=%d celdas, fill >=%.2f, aspect <=%.1f'
          % (dict(current.as_dict()), shape['gridStep'], shape['minBlobCells'],
             shape['minFillRatio'], shape['maxAspectRatio']))

    captures, skipped = load_captures(args.captures, args.ball_globs, args.bg_globs,
                                      args.width, args.height)
    for name in skipped:
        print('  (ignorada, no casa con ball*/bg*: %s)' % name)
    balls = [c for c in captures if c.kind == 'ball']
    backgrounds = [c for c in captures if c.kind == 'bg']
    if not balls or not backgrounds:
        raise SystemExit('se necesitan capturas ball* y bg* en %s (ver docs/CALIBRACION_BALON.md)' % args.captures)
    widths = {c.y.shape[1] for c in captures}
    heights = {c.y.shape[0] for c in captures}
    print('Capturas: %d de balón, %d de fondo · %s px' % (len(balls), len(backgrounds),
          ' y '.join('%dx%d' % (w, h) for w in widths for h in heights)))
    for capture in captures:
        sample(capture, shape['gridStep'], args.clip_bottom)

    print('\n== Evaluación con los umbrales actuales ==')
    current_rows = evaluate_set(captures, current, shape, max(widths), args.hfov)
    print_table(current_rows)
    fp = sum(1 for row in current_rows if row['verdict'] == 'FALSO_POSITIVO')
    miss = sum(1 for row in current_rows if row['verdict'] == 'NO_DETECTA')
    current_ok = fp == 0 and miss == 0
    print('Actuales: %s (%d falsos positivos, %d sin detectar)'
          % ('UTILIZABLES con estas capturas' if current_ok else 'NO UTILIZABLES con estas capturas', fp, miss))
    if current_ok:
        margin_bg = bg_margin([c.sampled for c in backgrounds], current)
        margin_ball = ball_margin(balls, current, shape)
        print('Márgenes: fondo %d, balón %d; tolerancia bilateral %d niveles YUV'
              % (margin_bg, margin_ball, min(margin_bg, margin_ball)))

    search = dict(QUICK if args.quick else SEARCH)
    if args.search_y:
        search.update(SEARCH_Y)
    total = 1
    for values in search.values():
        total *= len(values)
    print('\n== Búsqueda de umbrales (%d candidatos) ==' % total)
    started = time.time()
    result = suggest(balls, backgrounds, current, shape, search,
                     on_progress=lambda done, tot: print('  %d/%d candidatos…' % (done, tot), flush=True))
    print('Búsqueda en %.1f s' % (time.time() - started))

    report = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'cfg': str(Path(args.cfg).resolve()),
              'captures_dir': str(Path(args.captures).resolve()),
              'shape': shape, 'current': {'thresholds': current.as_dict(), 'usable': current_ok,
                                          'false_positives': fp, 'missed': miss,
                                          'per_image': current_rows}}
    if result is None:
        print('NINGÚN candidato factible: revisa las capturas (¿balón muy oscuro/brillante?, '
              '¿fondo con naranja real?) o amplía la búsqueda con --search-y.')
        report['suggestion'] = None
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
        return 2

    thresholds = result['thresholds']
    print('Sugerencia: %s' % dict(thresholds.as_dict()))
    print('Márgenes: fondo %d, balón %d; tolerancia bilateral %d niveles YUV'
          % (result['bg_margin'], result['ball_margin'], result['score']))
    print('\n== Evaluación con la sugerencia ==')
    suggested_rows = evaluate_set(captures, thresholds, shape, max(widths), args.hfov)
    print_table(suggested_rows)
    report['suggestion'] = {'thresholds': thresholds.as_dict(), 'bg_margin': result['bg_margin'],
                            'ball_margin': result['ball_margin'], 'score': result['score'],
                            'candidates': result['candidates'], 'per_image': suggested_rows}

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('\nReporte: %s' % args.report)
    if args.apply:
        backup = apply_thresholds(args.cfg, thresholds)
        print('Copia de seguridad: %s' % backup)
        print('Aplicado a %s' % args.cfg)
        print('Siguiente paso: desplegar Config SIN recompilar (DeployCopaNAO.bat con '
              '«Compilar antes de desplegar» desmarcado) y verificar en el robot con los logs '
              '[CopaPenalty] — ver docs/CALIBRACION_BALON.md §5.')
    elif not current_ok:
        print('\nPara aplicar la sugerencia: añade --apply (crea respaldo .bak junto al .cfg).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
