"""Bounded local pipes to one original B-Human C++ runtime per robot."""
from collections import deque
import json
import math
import os
import queue
import subprocess
import threading
import time

from penalty.motion_contract import CYCLE_S, JOINT_NAMES, ROOT, SOURCE_COMMIT, executable_path

MAX_FRAME = 65536


def finite_array(value, size, label):
    if (not isinstance(value,(list,tuple)) or len(value)!=size or
            any(isinstance(n,bool) or not isinstance(n,(float,int)) or
                not math.isfinite(n) for n in value)):
        raise RuntimeError('%s debe contener %d números finitos' % (label,size))
    return value


def _reject_constant(value):
    raise ValueError('valor JSON no finito: '+value)


class BHumanBridge:
    def __init__(self, role, trial, seed=0, timeout=2., executable=None, command=None):
        if role not in ('striker','keeper') or not isinstance(trial,str) or not trial:
            raise ValueError('identidad de robot/ensayo inválida')
        self.role,self.trial,self.timeout = role,trial,float(timeout)
        if not math.isfinite(self.timeout) or self.timeout<=0:
            raise ValueError('timeout debe ser positivo y finito')
        self._incoming = queue.Queue(maxsize=8)
        self._outgoing = queue.Queue(maxsize=2)
        self._stderr = deque(maxlen=8)
        self._failure = None
        self._closed = False
        self._last_sequence = -1
        self._last_time = None
        if command is None:
            exe = executable or executable_path()
            if not exe.is_file():
                raise RuntimeError('Motor B-Human sin compilar. Ejecuta: '
                                   '.\\.venv\\Scripts\\python.exe tools/build_motion_runtime.py')
            command = [str(exe.resolve())]
        flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        self.process = subprocess.Popen(command,cwd=str(ROOT),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,creationflags=flags)
        self._reader = threading.Thread(target=self._read_stdout,daemon=True)
        self._err_reader = threading.Thread(target=self._read_stderr,daemon=True)
        self._writer = threading.Thread(target=self._write_stdin,daemon=True)
        for thread in (self._reader,self._err_reader,self._writer):
            thread.start()
        try:
            self.hello = self._read('HELLO')
            if (type(self.hello.get('protocol')) is not int or self.hello['protocol']!=1 or
                    self.hello.get('source_commit')!=SOURCE_COMMIT or
                    tuple(self.hello.get('joint_names',[]))!=JOINT_NAMES or
                    abs(float(self.hello.get('cycle_s',0))-CYCLE_S)>1e-9):
                raise RuntimeError('HELLO B-Human incompatible con el modelo/protocolo')
            self._send({'type':'INIT','protocol':1,'trial':trial,'role':role,'seed':int(seed)})
            ready = self._read('READY')
            if ready.get('trial')!=trial or ready.get('role')!=role:
                raise RuntimeError('INIT respondió con identidad incorrecta')
        except Exception:
            self.close()
            raise

    def _read_stdout(self):
        try:
            while True:
                raw = self.process.stdout.readline(MAX_FRAME+1)
                if not raw:
                    self._failure = 'El motor cerró su canal de salida'
                    return
                if len(raw)>MAX_FRAME or not raw.endswith(b'\n'):
                    raise ValueError('salida demasiado grande o incompleta')
                message = json.loads(raw.decode('utf-8'),parse_constant=_reject_constant)
                if not isinstance(message,dict):
                    raise ValueError('la salida JSON debe ser un objeto')
                self._incoming.put_nowait(message)
        except (OSError,ValueError,queue.Full) as error:
            self._failure = 'Salida B-Human inválida: '+str(error)

    def _read_stderr(self):
        try:
            for raw in iter(lambda:self.process.stderr.readline(4097),b''):
                self._stderr.append(raw[:4096].decode('utf-8',errors='replace').strip())
        except OSError:
            pass

    def _write_stdin(self):
        try:
            while True:
                raw = self._outgoing.get()
                if raw is None:
                    return
                self.process.stdin.write(raw)
                self.process.stdin.flush()
        except (OSError,ValueError) as error:
            self._failure = 'Canal B-Human cerrado: '+str(error)

    def _send(self, message):
        raw = (json.dumps(message,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')
        if len(raw)>MAX_FRAME:
            raise RuntimeError('mensaje hacia B-Human demasiado grande')
        if self._closed or self.process.poll() is not None:
            raise RuntimeError('el proceso B-Human no está disponible')
        try:
            self._outgoing.put_nowait(raw)
        except queue.Full:
            raise RuntimeError('el motor no consume sus mensajes') from None

    def _read(self, expected):
        deadline = time.monotonic()+self.timeout
        while True:
            try:
                message = self._incoming.get(timeout=min(.05,max(.001,deadline-time.monotonic())))
                break
            except queue.Empty:
                if self._failure or time.monotonic()>=deadline:
                    context = ' | '.join(self._stderr)
                    raise RuntimeError('%s: %s%s' % (self.role,self._failure or
                        'timeout esperando '+expected,(' — '+context) if context else '')) from None
        if message.get('type')=='ERROR':
            raise RuntimeError('%s: %s' % (self.role,message.get('message',message)))
        if message.get('type')!=expected:
            raise RuntimeError('se esperaba %s y llegó %s' % (expected,message.get('type')))
        return message

    def step(self, sensors, request, sequence, time_ms):
        if type(sequence) is not int or sequence!=self._last_sequence+1:
            raise RuntimeError('secuencia de entrada inválida')
        if type(time_ms) is not int or time_ms<0 or (self._last_time is not None and time_ms-self._last_time!=12):
            raise RuntimeError('el reloj motriz debe avanzar exactamente 12 ms')
        for name,size in (('angles',26),('velocity',26),('gyro',3),('acc',3),
                           ('orientation',4),('fsr',8)):
            finite_array(sensors.get(name),size,'sensores '+name)
        if request.get('motion') not in ('stand','walk','kick','block','getup'):
            raise RuntimeError('petición de movimiento inválida')
        self._send({'type':'STEP','trial':self.trial,'sequence':sequence,
                    'time_ms':time_ms,'sensors':sensors,'request':request})
        output = self._read('OUTPUT')
        if (output.get('trial')!=self.trial or type(output.get('sequence')) is not int or
                output['sequence']!=sequence):
            raise RuntimeError('respuesta B-Human de otro ensayo o secuencia')
        finite_array(output.get('angles'),26,'ángulos de salida')
        finite_array(output.get('stiffness'),26,'rigidez de salida')
        finite_array(output.get('odometry'),3,'odometría')
        if any(s<0 or s>100 for s in output['stiffness']):
            raise RuntimeError('rigidez fuera del rango 0..100')
        if type(output.get('busy')) is not bool or type(output.get('kick_completed')) is not bool:
            raise RuntimeError('estado motriz inválido')
        self._last_sequence,self._last_time = sequence,time_ms
        return output

    def close(self):
        if self._closed:
            return
        try:
            if self.process.poll() is None:
                try:
                    self._send({'type':'SHUTDOWN','trial':self.trial})
                except (RuntimeError,OSError):
                    # Saturated/broken pipes must never skip child collection.
                    pass
                try:
                    self.process.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=.5)
        finally:
            self._closed = True
            try:
                self._outgoing.put_nowait(None)
            except queue.Full:
                pass
            for thread in (self._writer,self._reader,self._err_reader):
                thread.join(timeout=.2)
            for pipe in (self.process.stdin,self.process.stdout,self.process.stderr):
                try:
                    pipe.close()
                except OSError:
                    pass
