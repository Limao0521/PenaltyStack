"""Baseline ligero para la pelota naranja en RGB intercalado.

Adaptación conceptual de los candidatos por color de TriondaBallSpotsProvider.
El umbral y la distancia necesitan calibración con imágenes de cada NAO.
"""
from __future__ import division

import math


def detect_orange_ball(rgb, width, height, stride=2, min_pixels=10):
    """Devuelve (distancia_m, bearing_rad, n_pixeles) o None.

    Acepta bytes RGB, 3 por píxel. Usa componentes 8-conexas en la rejilla
    reducida; el mayor candidato se filtra por compacidad de caja.
    """
    if len(rgb) != width * height * 3:
        raise ValueError('RGB buffer size mismatch')
    cells = set()
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            i = (y * width + x) * 3
            r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
            if not isinstance(r, int):  # Python 2 str indexing
                r, g, b = ord(r), ord(g), ord(b)
            if 115 <= r and 45 <= g <= 200 and b <= 120 and r > g * 1.18 and g > b * 1.15:
                cells.add((x // stride, y // stride))
    best = None
    while cells:
        seed = cells.pop()
        queue = [seed]
        group = [seed]
        while queue:
            cx, cy = queue.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in cells:
                        cells.remove(neighbor)
                        queue.append(neighbor)
                        group.append(neighbor)
        if best is None or len(group) > len(best):
            best = group
    if not best or len(best) < min_pixels:
        return None
    xs = [p[0] for p in best]
    ys = [p[1] for p in best]
    box_w = (max(xs) - min(xs) + 1) * stride
    box_h = (max(ys) - min(ys) + 1) * stride
    if box_w / float(box_h) > 1.8 or box_h / float(box_w) > 1.8:
        return None
    cx = (sum(xs) / float(len(xs)) + 0.5) * stride
    diameter_px = (box_w + box_h) / 2.0
    focal_px = (width / 2.0) / math.tan(math.radians(54.7) / 2.0)
    distance = 0.065 * focal_px / max(1.0, diameter_px)
    bearing = -math.atan((cx - width / 2.0) / focal_px)
    return distance, bearing, len(best)
