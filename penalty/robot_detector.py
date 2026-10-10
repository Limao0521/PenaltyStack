"""Bounded local image IPC for the original B-Human network; Python 2.7.

No model downloads, sockets or actuator calls occur at runtime.
"""
from __future__ import division

import base64
from collections import deque
import hashlib
import io
import json
import os
import subprocess
import threading
import time
try:
    import queue
except ImportError:
    import Queue as queue

from penalty.nao_vision import monotonic_seconds
from penalty.perception import finite, vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_COMMIT = '2360ab1100ad28f70e190c893f6b095651ff0bbc'
VERSION = 'bhuman_robot_pixels_v1'
MAX_INPUT, MAX_OUTPUT = 1300000, 65536
MODEL_RELATIVE = 'Config/NeuralNets/RobotDetector/4_anchor_boxes_model_no_activation_20230629-220730.onnx'


def executable_path():
    suffix = 'Release/bhuman_vision.exe' if os.name == 'nt' else 'bhuman_vision'
    return os.path.join(ROOT, 'native/bhuman_motion/build/vision', suffix)


def model_path():
    return os.path.join(ROOT, 'heritage/bhuman_robot_perception', MODEL_RELATIVE)


def reject_constant(value):
    raise ValueError('nonfinite JSON: ' + value)


class RobotDetector(object):
    def __init__(self, executable=None, model=None, timeout=.25, command=None):
        self.timeout = float(timeout)
        if not finite(self.timeout) or not 0 < self.timeout <= 5.:
            raise ValueError('detector timeout must be finite, positive, at most 5 s')
        self._incoming, self._outgoing = queue.Queue(2), queue.Queue(2)
        self._failure, self._closed, self._sequence = None, False, 0
        self._stderr = deque(maxlen=4)
        self.process = None
        self._threads = []
        model = model or model_path()
        with open(os.path.join(ROOT, 'heritage/bhuman_robot_perception/manifest.json')) as stream:
            original = json.load(stream)
        with open(model, 'rb') as stream:
            self.model_sha256 = hashlib.sha256(stream.read()).hexdigest()
        if original['commit'] != SOURCE_COMMIT or self.model_sha256 != original['files'][MODEL_RELATIVE]:
            raise ValueError('original robot model/revision checksum differs')
        command = command or [executable or executable_path(), model]
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, bufsize=0, creationflags=flags)
            # io.open(fileno) also works with Python 2's legacy Popen files.
            self._stdout = io.open(self.process.stdout.fileno(), 'rb', buffering=MAX_OUTPUT, closefd=False)
            self._stdin = io.open(self.process.stdin.fileno(), 'wb', buffering=MAX_INPUT, closefd=False)
            for target in (self._read_stdout, self._write_stdin, self._read_stderr):
                thread = threading.Thread(target=target)
                thread.daemon = True
                self._threads.append(thread)
                thread.start()
            self.hello = self._read('HELLO', timeout=5.)
            if (self.hello.get('protocol') != VERSION or self.hello.get('source_commit') != SOURCE_COMMIT or
                    self.hello.get('model_sha256') != self.model_sha256 or self.hello.get('actuators') is not False):
                raise RuntimeError('incompatible robot-detector provenance')
        except Exception:
            self.close()
            raise

    def _read_stdout(self):
        try:
            while True:
                raw = self._stdout.readline(MAX_OUTPUT+1)
                if not raw or len(raw) > MAX_OUTPUT or not raw.endswith(b'\n'):
                    raise ValueError('closed, oversized or incomplete detector output')
                message = json.loads(raw.decode('utf-8'), parse_constant=reject_constant)
                if not isinstance(message, dict):
                    raise ValueError('detector output must be an object')
                self._incoming.put_nowait(message)
        except (OSError, IOError, ValueError, queue.Full) as error:
            self._failure = str(error)

    def _read_stderr(self):
        try:
            for raw in iter(lambda: self.process.stderr.readline(4097), b''):
                self._stderr.append(raw[:4096].decode('utf-8', 'replace').strip())
        except (OSError, IOError, ValueError):
            pass

    def _write_stdin(self):
        try:
            while True:
                raw = self._outgoing.get()
                if raw is None:
                    return
                self._stdin.write(raw)
                self._stdin.flush()
        except (OSError, IOError, ValueError) as error:
            self._failure = str(error)

    def _read(self, expected, timeout=None):
        deadline = monotonic_seconds() + (self.timeout if timeout is None else timeout)
        while True:
            try:
                message = self._incoming.get(timeout=min(.025, max(.001, deadline-monotonic_seconds())))
                break
            except queue.Empty:
                if self._failure or monotonic_seconds() >= deadline:
                    raise RuntimeError(self._failure or 'robot detector timeout')
        if message.get('type') == 'ERROR':
            raise RuntimeError(message.get('message', 'robot detector error'))
        if message.get('type') != expected:
            raise RuntimeError('unexpected detector response: ' + str(message.get('type')))
        return message

    def detect(self, pixels, width, height, capture_s, pixel_format='rgb', camera='upper'):
        capture_s = vector((capture_s,), 1)[0]
        if ((width, height) not in ((640, 480), (320, 240)) or camera not in ('upper', 'lower') or pixel_format not in ('rgb', 'yuyv') or
                len(pixels) != width*height*(3 if pixel_format == 'rgb' else 2)):
            raise ValueError('calibrated upper VGA/QVGA RGB or YUYV image required')
        if self._closed or self.process.poll() is not None:
            raise RuntimeError('robot detector is closed')
        try:
            raw = (json.dumps({'type': 'IMAGE', 'version': VERSION, 'sequence': self._sequence,
                               'capture_s': capture_s, 'width': width, 'height': height,
                               'format': pixel_format, 'camera': camera, 'pixels_b64': base64.b64encode(pixels).decode('ascii')},
                              separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')
            if len(raw) > MAX_INPUT:
                raise RuntimeError('detector input too large')
            self._outgoing.put_nowait(raw)
            output = self._read('DETECTIONS')
            if (output.get('version') != VERSION or output.get('sequence') != self._sequence or
                    output.get('capture_s') != capture_s or output.get('actuators') is not False or
                    output.get('camera') != camera or (output.get('width'), output.get('height')) != (width, height)):
                raise RuntimeError('detector response belongs to another frame')
            if not isinstance(output.get('boxes'), list) or len(output['boxes']) > 320:
                raise RuntimeError('invalid detector box list')
            for box in output['boxes']:
                vector(box['box'], 4)
                confidence = vector((box['confidence'],), 1)[0]
                if not .6 <= confidence <= 1.:
                    raise RuntimeError('invalid detector confidence')
            self._sequence += 1
            output['model_sha256'] = self.model_sha256
            return output
        except Exception:
            self.close()  # Broken/full IPC must still reap this child.
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
            deadline = monotonic_seconds()+.25
            while process.poll() is None and monotonic_seconds() < deadline:
                time.sleep(.01)
            if process.poll() is None:
                process.kill()
            process.wait()
        try:
            self._outgoing.put_nowait(None)
        except queue.Full:
            # Wake the writer after it finishes a pending broken-pipe write.
            try:
                self._outgoing.get_nowait()
                self._outgoing.put_nowait(None)
            except (queue.Full, queue.Empty):
                pass
        for thread in self._threads:
            thread.join(.2)
        for pipe in (getattr(self, '_stdin', None), getattr(self, '_stdout', None),
                     process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except (OSError, IOError, ValueError):
                    pass
