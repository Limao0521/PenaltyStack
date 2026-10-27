#!/usr/bin/env python3
"""Digital twin of the Copa penalty behavior.

The robot sends what it believes and decides (CopaTwinShared.h) after each
upper-camera frame. This tool draws it on a top-down map of the Copa field:
goal posts it saw, expected penalty mark, the robot's soles and trail, the
ball and the spot it memorised on the mark, the pre-kick pose it aligns at,
where it is walking to right now, the aim line and the keeper. Each run is
saved as JSONL so it can be replayed afterwards.

    live:    copa_twin.py --ip 192.168.49.7
    replay:  copa_twin.py --replay runs/twin/<run>.jsonl [--video out.mp4]

The camera viewer (copa_camera_viewer.py) shows the same map next to the
cameras and records runs automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # parsing and recording work without OpenCV
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
TWIN_FORMAT = struct.Struct('<IIBBB' + 'Bfff' + 'Bff' * 4 + 'B' + 'Bff' * 2 + 'b' + 'Bfff' * 2 + 'fff' + 'fB' + 'B' + 'ffffff')
STATES = {0: 'ARMED', 1: 'RUNNING', 2: 'STOPPED'}
STRIKER_PHASES = {0: 'tomar balon = marca', 1: 'mirar porteria y decidir', 2: 'alinear (sin tocar)', 3: 'patear',
                  4: 'armado: buscar balon', 5: 'armado: confirmar postes', 6: 'armado: listo'}
KEEPER_PHASES = {0: 'watch', 1: 'walk', 2: 'block left', 3: 'block right'}

# Copa NAO CCM MX 2026 geometry (mm) and B-Human RobotDimensions (soles).
GOAL_WIDTH, PENALTY_DISTANCE, FIELD = 1500.0, 1300.0, 2400.0
BALL_RADIUS = 32.5
SOLE = {'hip': 50.0, 'front': 110.0, 'back': 50.0, 'inner': 38.77, 'outer': 50.02}
MIN_CLEARANCE = 40.0


def parse(raw: bytes) -> dict:
    """Twin bytes (after the twinPresent flag) to a dict; see CopaCameraStreamer.h."""
    v = list(TWIN_FORMAT.unpack(raw))
    out = {'time': v.pop(0), 'run': v.pop(0), 'role': v.pop(0), 'state': v.pop(0), 'phase': v.pop(0)}

    def pose():
        valid, x, y, r = (v.pop(0) for _ in range(4))
        return {'valid': bool(valid), 'x': x, 'y': y, 'rotation': r}

    def point():
        valid, x, y = (v.pop(0) for _ in range(3))
        return {'valid': bool(valid), 'x': x, 'y': y}

    out['robot'] = pose()
    for name in ('ball', 'mark', 'left_post', 'right_post'):
        out[name] = point()
    out['goal_inferred'] = bool(v.pop(0))
    out['keeper'] = point()
    out['target'] = point()
    out['side'] = v.pop(0)
    out['pre_kick'] = pose()
    out['walk_target'] = pose()
    (out['foot_clearance'], out['min_foot_clearance'], out['ball_from_mark'], out['heading_correction'],
     out['posts_matched'], mark_seen, out['box_max_back'], out['box_max_lateral'], out['line_distance'],
     out['goal_heading'], out['pitch_offset'], out['track_shift']) = v
    out['mark_seen'] = bool(mark_seen)
    return out


class TwinRecorder:
    """One JSONL file per run (the robot increments `run` at every start)."""

    def __init__(self, directory: Path = ROOT / 'runs' / 'twin'):
        self.directory = Path(directory)
        self.file = None
        self.run = None
        self.last_time = None
        self.path = None

    def add(self, twin: dict) -> None:
        if twin['run'] == 0:
            return
        if twin['state'] == 2 and twin['run'] == self.run:  # "mode stop": the run is over
            self.close()
            return
        if twin['state'] == 2:
            return
        if twin['run'] != self.run:
            self.close()
            self.directory.mkdir(parents=True, exist_ok=True)
            role = 'keeper' if twin['role'] else 'striker'
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.path = self.directory / f'twin_{stamp}_{role}_run{twin["run"]}.jsonl'
            self.file = self.path.open('w')
            self.run = twin['run']
            self.last_time = None
            print(f'  [twin] grabando {self.path}')
        if twin['time'] != self.last_time:
            self.last_time = twin['time']
            self.file.write(json.dumps({**twin, 'pc_time': time.time()}) + '\n')
            self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None
            print(f'  [twin] corrida guardada: {self.path}')


# ── Drawing ───────────────────────────────────────────────────────────────────

class TwinMap:
    """Top-down map in the run's start frame: x forward (up in the image), y left."""

    def __init__(self, size: int = 520, x_range=(-600.0, 2100.0)):
        self.size = size
        self.x0, self.x1 = x_range
        self.scale = size / (self.x1 - self.x0)
        self.trail: list[tuple[float, float]] = []
        self.run = None

    def px(self, x: float, y: float) -> tuple[int, int]:
        return (int(round(self.size / 2 - y * self.scale)), int(round(self.size - (x - self.x0) * self.scale)))

    def _feet(self, img, pose, color, thickness=1):
        c, s = math.cos(pose['rotation']), math.sin(pose['rotation'])
        for sign in (1, -1):
            inner, outer = sign * (SOLE['hip'] - SOLE['inner']), sign * (SOLE['hip'] + SOLE['outer'])
            corners = [(-SOLE['back'], inner), (SOLE['front'], inner), (SOLE['front'], outer), (-SOLE['back'], outer)]
            pts = [self.px(pose['x'] + c * a - s * b, pose['y'] + s * a + c * b) for a, b in corners]
            cv2.polylines(img, [np.array(pts, np.int32)], True, color, thickness, cv2.LINE_AA)
        tip = self.px(pose['x'] + c * 160, pose['y'] + s * 160)
        cv2.arrowedLine(img, self.px(pose['x'], pose['y']), tip, color, thickness, cv2.LINE_AA, tipLength=0.3)

    def draw(self, twin: dict) -> np.ndarray:
        img = np.full((self.size, self.size + 340, 3), 28, np.uint8)
        field = img[:, :self.size]
        if twin['run'] != self.run:
            self.trail, self.run = [], twin['run']
        robot = twin['robot']
        if not self.trail or math.hypot(robot['x'] - self.trail[-1][0], robot['y'] - self.trail[-1][1]) > 5:
            self.trail.append((robot['x'], robot['y']))

        for g in range(-2000, 3000, 500):  # 0.5 m grid
            cv2.line(field, self.px(g, -1500), self.px(g, 1500), (45, 45, 45), 1)
            cv2.line(field, self.px(-1000, g), self.px(2500, g), (45, 45, 45), 1)

        left, right = twin['left_post'], twin['right_post']
        if left['valid'] and right['valid']:
            L, R = np.array([left['x'], left['y']]), np.array([right['x'], right['y']])
            center = (L + R) / 2
            along = (L - R) / max(np.linalg.norm(L - R), 1e-6)
            normal = np.array([along[1], -along[0]])
            if np.dot(normal, np.array([robot['x'], robot['y']]) - center) < 0:
                normal = -normal
            half = FIELD / 2
            corners = [center + along * half, center - along * half,
                       center - along * half + normal * FIELD, center + along * half + normal * FIELD]
            cv2.polylines(field, [np.array([self.px(*p) for p in corners], np.int32)], True, (90, 140, 90), 1, cv2.LINE_AA)
            cv2.line(field, self.px(*L), self.px(*R), (235, 235, 235), 2, cv2.LINE_AA)
            for post in (L, R):
                cv2.circle(field, self.px(*post), 6, (255, 120, 30), -1, cv2.LINE_AA)
            expected = center + normal * PENALTY_DISTANCE
            cv2.circle(field, self.px(*expected), max(2, int(50 * self.scale)), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(field, 'marca', (self.px(*expected)[0] + 8, self.px(*expected)[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)
            label = ('porteria desde la marca (sin confirmar)' if twin['goal_inferred']
                     else f'porteria confirmada, giro {math.degrees(twin["heading_correction"]):+.1f} deg')
            cv2.putText(field, label, (self.px(*L)[0], self.px(*L)[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 160, 80), 1, cv2.LINE_AA)

        if len(self.trail) > 1:
            cv2.polylines(field, [np.array([self.px(*p) for p in self.trail], np.int32)], False, (150, 150, 150), 1, cv2.LINE_AA)

        mark, ball, target = twin['mark'], twin['ball'], twin['target']
        if mark['valid'] and twin['line_distance'] > 0 and twin['role'] == 0:
            # Ball frame: origin on the mark, x towards the goal. Soles never past the red line;
            # the robot's origin stays inside the green box while it adjusts.
            hc, hs = math.cos(twin['goal_heading']), math.sin(twin['goal_heading'])
            to_start = lambda a, b: (mark['x'] + hc * a - hs * b, mark['y'] + hs * a + hc * b)
            half = twin['box_max_lateral'] + 120
            cv2.line(field, self.px(*to_start(-twin['line_distance'], half)), self.px(*to_start(-twin['line_distance'], -half)),
                     (60, 60, 230), 1, cv2.LINE_AA)
            box = [to_start(-twin['box_max_back'], twin['box_max_lateral']), to_start(-twin['line_distance'] - SOLE['front'], twin['box_max_lateral']),
                   to_start(-twin['line_distance'] - SOLE['front'], -twin['box_max_lateral']), to_start(-twin['box_max_back'], -twin['box_max_lateral'])]
            cv2.polylines(field, [np.array([self.px(*p) for p in box], np.int32)], True, (80, 160, 80), 1, cv2.LINE_AA)
        if target['valid'] and mark['valid']:
            cv2.line(field, self.px(mark['x'], mark['y']), self.px(target['x'], target['y']), (60, 60, 230), 2, cv2.LINE_AA)
            t = self.px(target['x'], target['y'])
            cv2.drawMarker(field, t, (60, 60, 230), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        if twin['keeper']['valid']:
            k = self.px(twin['keeper']['x'], twin['keeper']['y'])
            cv2.circle(field, k, 9, (200, 80, 200), 2, cv2.LINE_AA)
            cv2.putText(field, 'K', (k[0] - 4, k[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 80, 200), 1, cv2.LINE_AA)
        if mark['valid']:
            cv2.circle(field, self.px(mark['x'], mark['y']), max(2, int(BALL_RADIUS * self.scale)) + 3, (210, 210, 210), 1, cv2.LINE_AA)
        if ball['valid']:
            cv2.circle(field, self.px(ball['x'], ball['y']), max(3, int(BALL_RADIUS * self.scale)), (0, 140, 255), -1, cv2.LINE_AA)

        if twin['pre_kick']['valid']:
            self._feet(field, twin['pre_kick'], (230, 210, 60), 1)
        if twin['walk_target']['valid']:
            w = twin['walk_target']
            cv2.arrowedLine(field, self.px(robot['x'], robot['y']), self.px(w['x'], w['y']), (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.2)
        clearance = twin['foot_clearance']
        self._feet(field, robot, (80, 80, 255) if 0 < clearance < MIN_CLEARANCE else (240, 240, 240), 2)

        role = 'PORTERO' if twin['role'] else 'TIRADOR'
        phases = KEEPER_PHASES if twin['role'] else STRIKER_PHASES
        side = {1: 'izquierda', -1: 'derecha', 0: 'centro/recto'}.get(twin['side'], '?')
        stopped = twin['state'] == 2
        goal_text = (f'porteria: {twin["posts_matched"]} postes confirman, giro {math.degrees(twin["heading_correction"]):+.1f} deg'
                     if twin['posts_matched'] else 'porteria: desde la marca (sin postes)')
        lines = [(f'{role}  run {twin["run"]}', (255, 255, 255)),
                 (f'estado: {STATES.get(twin["state"], twin["state"])}', (200, 200, 200)),
                 ('fase: detenido (armar de nuevo)' if stopped else f'fase: {phases.get(twin["phase"], twin["phase"])}', (0, 220, 255)),
                 (f'marca: {"balon visto" if twin["mark_seen"] else "supuesta 20 cm"}', (210, 210, 210)),
                 (goal_text, (255, 160, 80)),
                 (f'portero: {"visto" if twin["keeper"]["valid"] else "no"}', (200, 80, 200)),
                 (f'tiro: {side}' if target['valid'] else 'tiro: sin decidir', (90, 90, 240)),
                 ('', None),
                 (f'balon-pie ahora: {clearance:5.0f} mm', (80, 80, 255) if 0 < clearance < MIN_CLEARANCE else (220, 220, 220)),
                 (f'balon-pie minimo: {twin["min_foot_clearance"]:5.0f} mm',
                  (80, 80, 255) if 0 < twin['min_foot_clearance'] < MIN_CLEARANCE else (220, 220, 220)),
                 (f'balon movido: {twin["ball_from_mark"]:5.0f} mm', (80, 80, 255) if twin['ball_from_mark'] > 30 else (220, 220, 220)),
                 (f'calibracion rango: {math.degrees(twin["pitch_offset"]):+.1f} deg', (220, 220, 220)),
                 (f'correccion en vivo: {twin["track_shift"]:5.0f} mm', (220, 220, 220)),
                 ('', None),
                 ('blanco: robot   cian: pose previa', (170, 170, 170)),
                 ('amarillo: a donde camina', (0, 220, 255)),
                 ('rojo: linea de tiro / linea del balon', (90, 90, 240)),
                 ('verde: zona permitida para ajustarse', (80, 160, 80))]
        for i, (text, color) in enumerate(lines):
            if color:
                cv2.putText(img, text, (self.size + 10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return img


# ── Live and replay ───────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summary(frames: list[dict]) -> dict:
    last = frames[-1]
    kicks = [f for f in frames if f['role'] == 0 and f['phase'] == 3]
    decided = [f for f in frames if f['target']['valid']]
    decision = decided[-1] if decided else last
    return {'frames': len(frames), 'duration_s': round((frames[-1]['time'] - frames[0]['time']) / 1000, 2),
            'side': decision['side'], 'posts_matched': decision['posts_matched'],
            'heading_correction_deg': round(math.degrees(decision['heading_correction']), 1),
            'keeper_seen': decision['keeper']['valid'],
            'min_foot_clearance_mm': round(decision['min_foot_clearance'], 1),
            'ball_moved_before_kick_mm': round(kicks[0]['ball_from_mark'], 1) if kicks else None}


def replay(path: Path, video: Path | None, speed: float) -> None:
    frames = load(path)
    twin_map = TwinMap()
    writer = None
    if video:
        first = twin_map.draw(frames[0])
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (first.shape[1], first.shape[0]))
        twin_map = TwinMap()
    start = frames[0]['time']
    next_video = 0.0
    for i, frame in enumerate(frames):
        img = twin_map.draw(frame)
        t = (frame['time'] - start) / 1000
        cv2.putText(img, f't={t:5.1f}s', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if writer:
            while next_video <= t:
                writer.write(img)
                next_video += 1 / 30
        else:
            cv2.imshow('Copa twin (replay)', img)
            delay = (frames[i + 1]['time'] - frame['time']) / speed if i + 1 < len(frames) else 0
            if cv2.waitKey(max(1, int(delay))) & 0xFF in (ord('q'), 27):
                break
    if writer:
        writer.release()
        print(f'video: {video}')
    else:
        cv2.waitKey(0)
    print(json.dumps(summary(frames)))


def live(ip: str, port: int) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location('viewer', Path(__file__).with_name('copa_camera_viewer.py'))
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    sock = viewer.connect(ip, port)
    twin_map, recorder = TwinMap(), TwinRecorder()
    try:
        while True:
            *_, copa = viewer.read_frame(sock)
            twin = copa and copa.get('twin')
            if twin:
                recorder.add(twin)
                cv2.imshow('Copa twin', twin_map.draw(twin))
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        recorder.close()
        sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ip', default='192.168.49.7')
    parser.add_argument('--port', type=int, default=7777, help='upper camera stream (carries the twin)')
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--video', type=Path, help='with --replay: write an MP4 instead of a window')
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()
    if args.replay:
        replay(args.replay, args.video, args.speed)
    else:
        live(args.ip, args.port)


if __name__ == '__main__':
    main()
