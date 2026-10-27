"""Trial lifecycle, time, FSM requests and UI command queue."""
import copy
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid

from penalty.control import Command, Observation, PenaltyController
from penalty.motion_contract import CYCLE_S, ROOT, SOURCE_COMMIT



# center: genuflectStand; left/right: keeper jumps (comparison only);
# defender_left/right: genuflectStandDefender, the RoboCup keeper block.
BLOCK_SIDES = ('center','left','right','defender_left','defender_right')

class SimulationSession:
    def __init__(self, simulator=None, record_dir=ROOT/'runs', mode='penalty'):
        if simulator is None:
            from penalty.physical_sim import PhysicalSimulator
            simulator = PhysicalSimulator()
        self.sim = simulator
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self.mode = mode
        self.role = 'striker'
        self.speed = 1.0
        self.state = 'PREPARED'
        self.message = 'Preparado. Pulsa Iniciar.'
        self.error = ''
        self.trial = None
        self.viewer = None
        self.states = []
        self.controllers = {r:PenaltyController(r) for r in ('striker','keeper')}
        self.commands = {r:Command('WAIT_FOR_START') for r in self.controllers}
        self.manual = {'motion':'stand', 'forward':0., 'lateral':0., 'turn':0.,
                       'kick_type':'forwardFastLeft', 'block_side':'center'}
        self._commands = queue.Queue()
        self._closing = threading.Event()
        self._lock = threading.Lock()
        self._cache = {}
        self._stream = None
        self._started_at = None
        self._frozen_penalty_time = None
        self._next_control = 0.
        self._stable_for = 0.
        self._paused_from = 'RUNNING'
        self._single_step = False
        self._initialized = False
        self._kick_requested = False
        self._kick_done = False
        self._keeper_block = None
        self._abort_at = None
        self._last_render = 0.
        self._publish()

    @property
    def penalty_time(self):
        if self._frozen_penalty_time is not None:
            return self._frozen_penalty_time
        if self._started_at is None:
            return 0.
        return max(0., min(28., float(self.sim.data.time)-self._started_at))

    def enqueue(self, action, value=None):
        self._commands.put((action, value))

    def key(self, keycode):
        keys = {82:'reset', 257:'repeat', 13:'repeat', 46:'step', 65:'abort'}
        if keycode == 32:
            self.enqueue('start' if self.snapshot()['state'] in
                         ('PREPARED','FINISHED','ABORTED') else 'pause')
        elif keycode in keys:
            self.enqueue(keys[keycode])

    def close(self):
        """Nonblocking request: the worker owns shutdown and child collection."""
        self._closing.set()

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._cache)

    def _publish(self):
        value = self.sim.snapshot()
        value.update(state=self.state, message=self.message, error=self.error,
                     trial=self.trial, penalty_time_s=self.penalty_time,
                     speed=self.speed, mode=self.mode, role=self.role)
        for role in self.controllers:
            value['robots'][role]['state'] = self.controllers[role].state
        with self._lock:
            self._cache = copy.deepcopy(value)

    def _record_start(self):
        if self.record_dir is None:
            return
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (self.record_dir/(self.trial+'.jsonl')).open('w',encoding='utf-8')
        hashes = {}
        for name in ('baseline/motion_manifest.json','models/bhuman/NaoV6H25.rsi3',
                     'models/bhuman/nao.xml','sim/scene.xml'):
            path = ROOT/name
            if path.is_file():
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self._write({'type':'METADATA', 'trial':self.trial, 'source_commit':SOURCE_COMMIT,
            'cycle_s':CYCLE_S, 'mode':self.mode, 'seed':getattr(self.sim,'seed',0),
            'hashes':hashes, 'units':{'position':'m','angle':'rad','fsr':'kg','effort':'N m'},
            'synthetic_inputs':['hand positions','motor currents/temperatures','joint wear'],
            'ideal_inputs':['scene ball/goal geometry','torso orientation'],
            'configuration':dict(self.manual),
            'engine_hello':{r:getattr(b,'hello',{}) for r,b in getattr(self.sim,'bridges',{}).items()}})

    def _write(self, record):
        if self._stream is not None:
            self._stream.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')

    def _record_end(self, reason):
        if self._stream is not None:
            self._write({'type':'SUMMARY','trial':self.trial,'reason':reason,
                         'penalty_time_s':self.penalty_time,'error':self.error,
                         'final':self.sim.snapshot(),'states':self.states})
            self._stream.close()
            self._stream = None

    def _prepare(self):
        self._record_end('reset')
        self.trial = uuid.uuid4().hex
        self.sim.reset(self.trial)
        self.controllers = {r:PenaltyController(r) for r in ('striker','keeper')}
        self.commands = {r:Command('WAIT_FOR_START') for r in self.controllers}
        self._started_at = None
        self._frozen_penalty_time = None
        self._next_control = self._stable_for = 0.
        self._kick_requested = self._kick_done = False
        self._keeper_block = None
        self._abort_at = None
        self._single_step = False
        self._initialized = True
        self.states = []
        self.state, self.error = 'PREPARED', ''
        self.message = 'Ensayo restablecido. Pulsa Iniciar.'
        self._record_start()

    def _start(self):
        if not self._initialized or self.state in ('FINISHED','ABORTED'):
            self._prepare()
        self.state = 'SETTLING'
        self.message = 'Estabilizando postura con B-Human…'

    @staticmethod
    def _validate_manual(value):
        request = dict(value)
        if request.get('motion') not in ('stand','walk','kick','block','getup','sit'):
            raise ValueError('movimiento manual desconocido')
        for name, bound in (('forward',.15),('lateral',.15),('turn',.35),
                             ('head_yaw',1.0),('head_pitch',.5)):
            number = float(request.get(name,0.))
            if not math.isfinite(number) or abs(number)>bound:
                raise ValueError('%s fuera del rango permitido' % name)
            request[name] = number
        if request.get('kick_type','forwardFastLeft') not in ('forwardFastLeft','forwardFastRight'):
            raise ValueError('tipo de patada desconocido')
        if request.get('block_side','center') not in BLOCK_SIDES:
            raise ValueError('lado de bloqueo desconocido')
        return request

    def command(self, action, value=None):
        """Worker-thread commands; tests may call directly without a GUI."""
        if action == 'speed':
            value = float(value)
            if value not in (.25,.5,1.,2.):
                raise ValueError('velocidad debe ser 0.25, 0.5, 1 o 2')
            self.speed = value
        elif action == 'role':
            if value not in self.controllers:
                raise ValueError('rol desconocido')
            self.role = value
        elif action == 'mode':
            if value not in ('penalty','manual'):
                raise ValueError('modo desconocido')
            if value != self.mode:
                self.mode = value
                self._prepare()
        elif action == 'manual':
            request = self._validate_manual(value)
            if self.mode != 'manual':
                raise ValueError('selecciona modo manual para pedir acciones')
            self.manual.update(request)
            self._kick_done = False
        elif action == 'start' and self.state in ('PREPARED','FINISHED','ABORTED'):
            self._start()
        elif action == 'reset':
            self._prepare()
        elif action == 'repeat':
            self._prepare()
            self._start()
        elif action == 'pause':
            if self.state in ('RUNNING','SETTLING'):
                self._paused_from, self.state = self.state, 'PAUSED'
                self.message = 'En pausa; los relojes y la física están detenidos.'
            elif self.state == 'PAUSED':
                self.state = self._paused_from
                self.message = 'Simulación reanudada.'
        elif action == 'step' and self.state in ('PREPARED','PAUSED'):
            if self.state == 'PREPARED':
                if not self._initialized:
                    self._prepare()
                self._paused_from = 'SETTLING'
                self.state = 'PAUSED'
            self._single_step = True
        elif action == 'abort' and self.state not in ('CLOSED','ERROR','ABORTED','FINISHED'):
            self._frozen_penalty_time = self.penalty_time
            for controller in self.controllers.values():
                controller.update(Observation(self.penalty_time,abort=True))
            if self._initialized:
                self.state = 'ABORTING'
                self._abort_at = self.sim.data.time
                self.message = 'ABORT: solicitando postura y finalización motriz.'
            else:
                self._finish('ABORTED')
        elif action == 'close':
            self.close()
        self._publish()

    def _requests(self):
        if self.mode == 'manual':
            selected = dict(self.manual)
            if selected['motion'] == 'kick' and self._kick_done:
                selected['motion'] = 'stand'
            return {r:selected if r == self.role else {'motion':'stand'} for r in self.controllers}
        elapsed = self.penalty_time
        if elapsed+1e-9 >= self._next_control:
            observations = self.sim.observation()
            for role, controller in self.controllers.items():
                values = observations[role]
                self.commands[role] = controller.update(Observation(elapsed,start=True,
                    ball_range=values['ball_range'],ball_bearing=values['ball_bearing'],
                    goal_bearing=values['goal_bearing'],motion_busy=values.get('motion_busy'),
                    motion_done=self._kick_done if role == 'striker' else False))
            self._next_control += .1
            pair = tuple(self.commands[r].state for r in self.controllers)
            if not self.states or self.states[-1]['states'] != list(pair):
                self.states.append({'time_s':elapsed,'states':list(pair)})
            if self.commands['striker'].kick:
                self._kick_requested = True
        requests = {}
        for role, command in self.commands.items():
            motion = 'walk' if command.forward or command.lateral or command.turn else 'stand'
            if role == 'striker' and self._kick_requested and not self._kick_done:
                motion = 'kick'
            elif role == 'keeper':
                # El portero no se lanza: en cuanto ve el balón se sienta con
                # el keyframe original sitDownKeeper (petición `sit`) y conserva
                # la guardia hasta terminar el ensayo o ABORT. Perder el balón
                # de vista o un cambio de rumbo no liberan la postura sentada.
                # La observación de suelo verdadero queda explícita en el
                # registro de la simulación. El lanzamiento original (block)
                # sigue disponible en modo manual para comparación.
                if command.state == 'BLOCK' and self._keeper_block is None:
                    self._keeper_block = command.block
                if self._keeper_block is not None:
                    motion = 'sit'
            requests[role] = {'motion':motion,'forward':command.forward*1000,
                'lateral':command.lateral*1000,'turn':command.turn,
                'head_yaw':command.head_yaw,'head_pitch':0.,
                'kick_type':self.manual['kick_type'],
                'block_side':self._keeper_block if role=='keeper' and self._keeper_block is not None else command.block}
        return requests

    def _finish(self, state='FINISHED'):
        if state == 'FINISHED':
            for controller in self.controllers.values():
                # Record STOP also for a shortened headless trial. Finishing
                # freezes physics and must not leave an active FSM in the UI.
                controller.update(Observation(self.penalty_time,abort=True))
            pair = [self.controllers[r].state for r in self.controllers]
            if not self.states or self.states[-1]['states'] != pair:
                self.states.append({'time_s':self.penalty_time,'states':pair})
        self.state = state
        self.message = 'Ensayo %s. Puedes repetir o reiniciar.' % (
            'abortado' if state == 'ABORTED' else 'terminado')
        self._record_end(state)

    def tick(self):
        if self._closing.is_set():
            self._record_end('closed')
            self.sim.close()
            self.state = 'CLOSED'
            self._publish()
            return
        while not self._commands.empty():
            action, value = self._commands.get_nowait()
            self.command(action,value)
        one = self._single_step
        self._single_step = False
        active = self._paused_from if self.state == 'PAUSED' and one else self.state
        if active not in ('SETTLING','RUNNING','ABORTING'):
            self._publish()
            return
        if active == 'RUNNING':
            requests = self._requests()
            if self.mode == 'manual':
                requests = copy.deepcopy(requests)
                for request in requests.values():
                    request['forward'] = request.get('forward',0)*1000
                    request['lateral'] = request.get('lateral',0)*1000
        else:
            requests = {r:{'motion':'stand'} for r in self.controllers}
        self.sim.step(requests)
        self._kick_done |= bool(self.sim.outputs[self.role if self.mode=='manual' else 'striker'].get('kick_completed'))
        self._write({'type':'FRAME','trial':self.trial,'state':active,
                     'penalty_time_s':self.penalty_time,'requests':requests,'world':self.sim.trace()})
        if active == 'SETTLING':
            self._stable_for = self._stable_for+self.sim.dt if self.sim.stable() else 0.
            if self.sim.data.time >= 2. and self._stable_for >= .5:
                self._started_at = self.sim.data.time
                if one:
                    self._paused_from = 'RUNNING'
                else:
                    self.state = 'RUNNING'
                self.message = 'Ensayo en marcha.'
            elif self.sim.data.time >= 5.:
                raise RuntimeError('B-Human no estabilizó ambos NAO en 5 s; revisa sensores/modelo')
        elif active == 'ABORTING':
            complete = all(not out.get('busy',False) and out.get('phase','stand') == 'stand'
                           for out in self.sim.outputs.values())
            if complete or self.sim.data.time-self._abort_at >= 2.:
                self._finish('ABORTED')
        elif active == 'RUNNING' and self.penalty_time >= 28.-1e-9:
            for controller in self.controllers.values():
                controller.update(Observation(28.,start=True))
            self._finish()
        self._publish()

    def run(self, realtime=True):
        """Background loop; GUI never waits for a native response."""
        try:
            while self.state != 'CLOSED':
                wall = time.monotonic()
                try:
                    if self.viewer is not None and not self.viewer.is_running():
                        self.close()
                    self.tick()
                    if self.viewer is not None and wall-self._last_render >= 1/30:
                        self.viewer.sync()
                        self._last_render = wall
                except Exception as error:
                    self.state,self.error = 'ERROR',str(error)
                    self.message = 'Ensayo detenido. Corrige el error y pulsa Reiniciar.'
                    self._record_end('error')
                    self.sim.close()
                    self._initialized = False
                    self._publish()
                wait = self.sim.dt/self.speed-(time.monotonic()-wall) if realtime else 0.
                if self.state not in ('RUNNING','SETTLING','ABORTING'):
                    wait = .02
                if wait>0:
                    self._closing.wait(wait)
        finally:
            self._record_end('closed')
            self.sim.close()
            if self.viewer is not None:
                self.viewer.close()
