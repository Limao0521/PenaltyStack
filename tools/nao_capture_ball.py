#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Capture orange-ball calibration images ON a NAO v6 (Python 2.7 / NAOqi 2.8).

Saves native YUV422 frames when the camera delivers them (exactly what
CopaOrangeBallPerceptor consumes), RGB .ppm otherwise. bhuman holds the camera through
V4L2, so stop it first — tools/capture_ball_calibration.py does this automatically:

    systemctl --user stop bhuman.service
    python nao_capture_ball.py --label ball_130 --cameras 0,1 --count 20
    systemctl --user start bhuman.service

Output: <out>/<label>/<label>_c<camera>_<index>.yuv|.ppm + capture_meta.json, fetched to
runs/ball_calibration/<robot>/<label>/ from the PC. Protocol and tuning workflow:
docs/CALIBRACION_BALON.md.
"""
from __future__ import print_function

import argparse
import json
import os
import socket
import sys
import time

# (colorSpace id pedido, formato esperado, bytes por píxel). El 12/108 es YUV422 nativo en
# NAOqi 2.x; 13/11 aparecen como RGB según la versión. Se prueban en orden y se valida el
# tamaño del buffer contra width*height*bytes: gana el primero coherente, sin confiar en
# constantes que varían entre releases de NAOqi.
COLORSPACE_CANDIDATES = ((12, 'yuv422', 2), (108, 'yuv422', 2), (13, 'rgb', 3), (11, 'rgb', 3))
RESOLUTIONS = {0: 'QQVGA 160x120', 1: 'QVGA 320x240', 2: 'VGA 640x480', 3: '4VGA 1280x960'}
CLIENT = 'PenaltyCalib'


def find_buffer(response, width, height, bytes_per_pixel):
    """El buffer de imagen de getImageRemote, sin depender del índice (varía entre docs)."""
    expected = width * height * bytes_per_pixel
    for item in response[2:]:
        # py2: el buffer llega como str (== bytes); py3: como bytes. Ambos pasan isinstance.
        if isinstance(item, (str, bytes)) and len(item) == expected:
            return item
    return None


def probe_format(video, camera_id, resolution, wanted):
    """Suscribe y valida un frame; devuelve (client_id, colorspace, fmt, w, h, buffer)."""
    for colorspace, fmt, bpp in COLORSPACE_CANDIDATES:
        if wanted != 'auto' and fmt != wanted:
            continue
        client = None
        try:
            client = video.subscribeCamera(CLIENT, camera_id, resolution, colorspace, 15)
            time.sleep(1.0)  # estabiliza exposición/balance de blancos automáticos
            response = video.getImageRemote(client)
            if response:
                width, height = int(response[0]), int(response[1])
                buffer_ = find_buffer(response, width, height, bpp)
                if buffer_ is not None:
                    return client, colorspace, fmt, width, height, buffer_
        except Exception as error:  # noqa: BLE001 - NAOqi lanza RuntimeError genéricos
            print('  colorspace %d (%s): %s' % (colorspace, fmt, error))
        if client is not None:
            try:
                video.unsubscribe(client)
            except Exception:  # noqa: BLE001
                pass
    return None


def save_frame(directory, name, fmt, width, height, buffer_):
    path = os.path.join(directory, name)
    with open(path, 'wb') as handle:
        if fmt == 'rgb':
            handle.write(b'P6\n%d %d\n255\n' % (width, height))
        handle.write(buffer_)
    return path


def capture_camera(video, camera_id, args, out_dir, shared_meta):
    probe = probe_format(video, camera_id, args.resolution, args.format)
    if probe is None:
        raise RuntimeError('camara %d: ningun formato valido (probados %s). '
                           '¿Esta bhuman corriendo? Detenlo: systemctl --user stop bhuman.service'
                           % (camera_id, [c for c, _, _ in COLORSPACE_CANDIDATES]))
    client, colorspace, fmt, width, height, first = probe
    ext = '.yuv' if fmt == 'yuv422' else '.ppm'
    shared_meta.update({'format': fmt, 'colorspace_requested': colorspace,
                        'width': width, 'height': height})
    print('camara %d: %s %dx%d, colorspace %d, %d bytes/frame'
          % (camera_id, fmt.upper(), width, height, colorspace, len(first)))
    try:
        for _ in range(args.warmup):
            video.getImageRemote(client)
            time.sleep(0.03)
        frames = []
        for index in range(args.count):
            response = video.getImageRemote(client)
            buffer_ = find_buffer(response, width, height, 2 if fmt == 'yuv422' else 3)
            if buffer_ is None:
                print('  frame %d sin buffer valido, se omite' % index)
                continue
            name = '%s_c%d_%03d%s' % (args.label, camera_id, index, ext)
            save_frame(out_dir, name, fmt, width, height, buffer_)
            frames.append({'file': name, 'camera': camera_id, 'index': index})
            print('  %s (%d/%d)' % (name, index + 1, args.count))
            time.sleep(args.interval_ms / 1000.0)
        return frames
    finally:
        video.unsubscribe(client)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Captura imagenes de calibracion del balon naranja.')
    parser.add_argument('--label', required=True,
                        help='etiqueta del lote: ball_020/ball_130/ball_200/bg_cancha/bg_distractores…')
    parser.add_argument('--cameras', default='0,1', help='0=superior, 1=inferior (omision: ambas)')
    parser.add_argument('--count', type=int, default=20, help='frames por camara')
    parser.add_argument('--interval-ms', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=15, help='frames descartados para estabilizar')
    parser.add_argument('--resolution', type=int, default=2,
                        help='2=VGA 640x480 (la del runtime B-Human); 1=QVGA')
    parser.add_argument('--format', choices=('auto', 'yuv422', 'rgb'), default='auto')
    parser.add_argument('--out', default='/home/nao/ball_calibration')
    args = parser.parse_args(argv)

    if args.resolution not in RESOLUTIONS:
        parser.error('resolution desconocida: %s (%s)' % (args.resolution, RESOLUTIONS))
    try:
        camera_ids = [int(part) for part in args.cameras.split(',') if part.strip()]
    except ValueError:
        parser.error('camaras invalidas: %s' % args.cameras)
    for camera_id in camera_ids:
        if camera_id not in (0, 1):
            parser.error('camara invalida: %d (0=superior, 1=inferior)' % camera_id)

    try:
        import qi
    except ImportError:
        print('sin modulo qi: este script corre EN el robot (NAOqi 2.8), no en el PC.', file=sys.stderr)
        return 1

    out_dir = os.path.join(args.out, args.label)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    session = qi.Session()
    try:
        session.connect('tcp://127.0.0.1:9559')
    except Exception as error:  # noqa: BLE001
        print('no hay conexion con NAOqi (127.0.0.1:9559): %s' % error, file=sys.stderr)
        return 1
    video = session.service('ALVideoDevice')

    meta = {'tool': 'nao_capture_ball.py', 'robot': socket.gethostname(), 'label': args.label,
            'date': time.strftime('%Y-%m-%dT%H:%M:%S'), 'cameras': camera_ids,
            'resolution_id': args.resolution, 'resolution': RESOLUTIONS[args.resolution],
            'interval_ms': args.interval_ms, 'warmup': args.warmup, 'frames': []}
    failed = []
    for camera_id in camera_ids:
        try:
            meta['frames'] += capture_camera(video, camera_id, args, out_dir, meta)
        except Exception as error:  # noqa: BLE001
            print('camara %d FALLO: %s' % (camera_id, error), file=sys.stderr)
            failed.append(camera_id)
    with open(os.path.join(out_dir, 'capture_meta.json'), 'wb') as handle:
        handle.write(json.dumps(meta, indent=2).encode('utf-8'))
    print('%d frames en %s%s' % (len(meta['frames']), out_dir,
                                 ' · CAMARAS FALLIDAS: %s' % failed if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
