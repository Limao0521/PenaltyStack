"""Deploy dialog for the Copa NAO CCM MX 2026 penalty routine (like Sabana Herons' DeployDialog).

Select the robots reachable over Ethernet, give each one its role (striker/keeper),
its tactical policies (RL versions) and player number, and deploy them in parallel.
Launch with DeployCopaNAO.bat (Windows) or `python deploy/deploy_dialog.py`.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
sys.path.insert(0, str(ROOT / 'tools'))

import check_module_graph  # noqa: E402
import robot_deploy  # noqa: E402

STATE = ROOT / 'deploy/.deploy_dialog_state.json'
ROLE_LABELS = {'striker': 'Tirador', 'keeper': 'Portero'}
START_LABELS = {'buttonOnly': 'Boton medio de la cabeza (tras el silbato)',
                'whistleOnly': 'Silbato del juez',
                'buttonOrWhistle': 'Boton o silbato (lo primero)'}
COLORS = {'ok': '#1a7f37', 'error': '#cf222e', 'unknown': '#8c959f', 'busy': '#9a6700'}


class RobotRow:
    def __init__(self, dialog, parent, row, robot, registry, saved):
        self.robot = robot
        self.selected = tk.BooleanVar(value=saved.get('selected', False))
        self.role = tk.StringVar(value=ROLE_LABELS.get(saved.get('role', 'striker'), 'Tirador'))
        self.striker = tk.StringVar(value=saved.get('striker_policy', registry['defaults']['striker']))
        self.keeper = tk.StringVar(value=saved.get('keeper_policy', registry['defaults']['keeper']))
        self.player = tk.IntVar(value=saved.get('player', row))
        ttk.Checkbutton(parent, variable=self.selected).grid(row=row, column=0, padx=4)
        ttk.Label(parent, text=robot.name, width=11).grid(row=row, column=1, sticky='w')
        ttk.Label(parent, text=robot.lan, width=14).grid(row=row, column=2, sticky='w')
        self.link = tk.Label(parent, text='● sin probar', fg=COLORS['unknown'], width=13, anchor='w')
        self.link.grid(row=row, column=3, sticky='w')
        role = ttk.Combobox(parent, textvariable=self.role, values=list(ROLE_LABELS.values()), width=9, state='readonly')
        role.grid(row=row, column=4, padx=2)
        striker = ttk.Combobox(parent, textvariable=self.striker, values=list(registry['striker']), width=20, state='readonly')
        striker.grid(row=row, column=5, padx=2)
        keeper = ttk.Combobox(parent, textvariable=self.keeper, values=list(registry['keeper']), width=18, state='readonly')
        keeper.grid(row=row, column=6, padx=2)
        for box, role_name in ((striker, 'striker'), (keeper, 'keeper')):
            box.bind('<<ComboboxSelected>>', lambda _event, b=box, r=role_name: dialog.describe(r, b.get()))
        ttk.Spinbox(parent, from_=1, to=20, textvariable=self.player, width=4).grid(row=row, column=7, padx=2)
        self.result = tk.Label(parent, text='', width=14, anchor='w')
        self.result.grid(row=row, column=8, sticky='w')

    def role_name(self):
        return 'keeper' if self.role.get() == ROLE_LABELS['keeper'] else 'striker'

    def state(self):
        return {'selected': self.selected.get(), 'role': self.role_name(), 'striker_policy': self.striker.get(),
                'keeper_policy': self.keeper.get(), 'player': self.player.get()}


class DeployDialog:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.busy = False
        self.registry = robot_deploy.policies()
        saved = self.load_state()
        root.title('PenaltyStack - Despliegue Copa NAO CCM MX 2026 (Tiro Penal)')
        root.geometry('1120x720')
        root.minsize(980, 600)

        general = ttk.LabelFrame(root, text='Configuracion general')
        general.pack(fill='x', padx=10, pady=(10, 4))
        self.config = tk.StringVar(value=saved.get('config', 'Release'))
        self.compile = tk.BooleanVar(value=saved.get('compile', True))
        self.team = tk.IntVar(value=saved.get('team', 49))
        self.start_mode = tk.StringVar(value=START_LABELS[saved.get('start_mode', 'buttonOnly')])
        self.wifi_off = tk.BooleanVar(value=saved.get('wifi_off', True))
        self.jumps = tk.BooleanVar(value=saved.get('jumps', True))
        self.restart = tk.BooleanVar(value=saved.get('restart', True))
        self.volume = tk.IntVar(value=saved.get('volume', 70))
        self.dry_run = tk.BooleanVar(value=False)
        ttk.Label(general, text='Compilacion').grid(row=0, column=0, sticky='w', padx=6, pady=3)
        ttk.Combobox(general, textvariable=self.config, values=robot_deploy.CONFIGS, width=9,
                     state='readonly').grid(row=0, column=1, sticky='w')
        ttk.Checkbutton(general, text='Compilar antes de desplegar', variable=self.compile).grid(row=0, column=2, sticky='w', padx=8)
        ttk.Label(general, text='Equipo #').grid(row=0, column=3, sticky='e', padx=6)
        ttk.Spinbox(general, from_=0, to=255, textvariable=self.team, width=5).grid(row=0, column=4, sticky='w')
        ttk.Label(general, text='Volumen %').grid(row=0, column=5, sticky='e', padx=6)
        ttk.Spinbox(general, from_=0, to=100, increment=5, textvariable=self.volume, width=5).grid(row=0, column=6, sticky='w')
        ttk.Label(general, text='Arranque').grid(row=1, column=0, sticky='w', padx=6, pady=3)
        ttk.Combobox(general, textvariable=self.start_mode, values=list(START_LABELS.values()), width=40,
                     state='readonly').grid(row=1, column=1, columnspan=2, sticky='w')
        ttk.Checkbutton(general, text='Apagar Wi-Fi (regla 2.7: solo cable)', variable=self.wifi_off).grid(row=1, column=3, columnspan=2, sticky='w', padx=6)
        ttk.Checkbutton(general, text='Portero con estiradas laterales', variable=self.jumps).grid(row=1, column=5, columnspan=2, sticky='w', padx=6)
        ttk.Checkbutton(general, text='Iniciar bhuman al terminar', variable=self.restart).grid(row=2, column=3, columnspan=2, sticky='w', padx=6)
        ttk.Checkbutton(general, text='Simulacro (no toca los robots)', variable=self.dry_run).grid(row=2, column=5, columnspan=2, sticky='w', padx=6)

        table = ttk.LabelFrame(root, text='Robots (red cableada)')
        table.pack(fill='x', padx=10, pady=4)
        for column, title in enumerate(('', 'Robot', 'IP LAN', 'Conexion', 'Rol', 'Politica tirador (RL)',
                                        'Politica portero', 'Jugador', 'Resultado')):
            ttk.Label(table, text=title, font=('Segoe UI', 9, 'bold')).grid(row=0, column=column, sticky='w', padx=2)
        rows = saved.get('robots', {})
        self.rows = [RobotRow(self, table, index + 1, robot, self.registry, rows.get(robot.name, {}))
                     for index, robot in enumerate(robot_deploy.robots())]

        self.description = ttk.Label(root, text='Selecciona una politica para ver su descripcion.', foreground='#57606a')
        self.description.pack(fill='x', padx=14)

        buttons = ttk.Frame(root)
        buttons.pack(fill='x', padx=10, pady=6)
        self.buttons = []
        for text, command in (('Comprobar conexion', self.check_links), ('Validar escenario', self.validate),
                              ('Compilar', self.build), ('Desplegar seleccionados', self.deploy),
                              ('Estado', lambda: self.on_selected('status')),
                              ('Detener bhuman', lambda: self.on_selected('stop')),
                              ('Iniciar bhuman', lambda: self.on_selected('start')),
                              ('Descargar logs', lambda: self.on_selected('logs'))):
            button = ttk.Button(buttons, text=text, command=command)
            button.pack(side='left', padx=3)
            self.buttons.append(button)
        ttk.Button(buttons, text='Seleccionar todos', command=lambda: self.select_all(True)).pack(side='right', padx=3)
        ttk.Button(buttons, text='Ninguno', command=lambda: self.select_all(False)).pack(side='right', padx=3)

        self.log = ScrolledText(root, height=16, font=('Consolas', 9), state='disabled')
        self.log.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        self.log.tag_config('error', foreground=COLORS['error'])
        self.log.tag_config('ok', foreground=COLORS['ok'])
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.pump)
        self.write('Robots leidos de robot/Config/Robots/*/network.cfg. Conecta el PC por Ethernet a la subred de los robots.')
        self.write('Al encender: boton del pecho = ponerse de pie; boton medio = arrancar; boton trasero = detener; '
                   'boton delantero = rearmar (mantener 2 s = cambiar rol).')
        self.check_links()

    # -- state ---------------------------------------------------------------------------
    def load_state(self):
        try:
            return json.loads(STATE.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}

    def save_state(self):
        start = next((k for k, v in START_LABELS.items() if v == self.start_mode.get()), 'buttonOnly')
        data = {'config': self.config.get(), 'compile': self.compile.get(), 'team': self.safe_int(self.team, 0),
                'start_mode': start, 'wifi_off': self.wifi_off.get(), 'jumps': self.jumps.get(),
                'restart': self.restart.get(), 'volume': self.safe_int(self.volume, 70),
                'robots': {row.robot.name: row.state() for row in self.rows}}
        try:
            STATE.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        except OSError:
            pass

    @staticmethod
    def safe_int(variable, default):
        try:
            return int(variable.get())
        except (tk.TclError, ValueError):
            return default

    def close(self):
        if self.busy and not messagebox.askyesno('Despliegue en curso', 'Hay una tarea en curso. Cerrar de todos modos?'):
            return
        self.save_state()
        self.root.destroy()

    # -- UI helpers ------------------------------------------------------------------------
    def write(self, text, tag=None):
        self.events.put(('log', text, tag))

    def pump(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == 'log':
                    _, text, tag = event
                    if tag is None:
                        tag = 'error' if ('error' in text.lower() or 'fallo' in text.lower()) else \
                              'ok' if ('DEPLOY_OK' in text or 'DESPLEGADO' in text) else None
                    self.log.configure(state='normal')
                    self.log.insert('end', time.strftime('%H:%M:%S ') + text + '\n', tag or ())
                    self.log.see('end')
                    self.log.configure(state='disabled')
                elif event[0] == 'link':
                    _, row, ok = event
                    row.link.configure(text='● responde' if ok else '● sin respuesta',
                                       fg=COLORS['ok' if ok else 'error'])
                elif event[0] == 'result':
                    _, row, text, kind = event
                    row.result.configure(text=text, fg=COLORS[kind])
                elif event[0] == 'busy':
                    self.busy = event[1]
                    for button in self.buttons:
                        button.configure(state='disabled' if self.busy else 'normal')
        except queue.Empty:
            pass
        self.root.after(100, self.pump)

    def describe(self, role, name):
        entry = self.registry[role].get(name, {})
        self.description.configure(text='%s %s [%s]: %s' % ('Tirador' if role == 'striker' else 'Portero', name,
                                                              entry.get('status', '?'), entry.get('description', '')))

    def select_all(self, value):
        for row in self.rows:
            row.selected.set(value)

    def selected(self):
        rows = [row for row in self.rows if row.selected.get()]
        if not rows:
            messagebox.showinfo('Sin robots', 'Selecciona al menos un robot.')
        return rows

    def background(self, work):
        if self.busy:
            return
        self.save_state()
        self.events.put(('busy', True))

        def run():
            try:
                work()
            except Exception as error:  # noqa: BLE001 - shown to the operator
                self.write('ERROR: %s' % error, 'error')
            finally:
                self.events.put(('busy', False))
        threading.Thread(target=run, daemon=True).start()

    def printer(self, prefix):
        return lambda line: self.write('[%s] %s' % (prefix, line))

    # -- actions ---------------------------------------------------------------------------
    def check_links(self):
        def work():
            with ThreadPoolExecutor(max_workers=max(1, len(self.rows))) as pool:
                for row, ok in zip(self.rows, pool.map(lambda r: robot_deploy.ping(r.robot), self.rows)):
                    self.events.put(('link', row, ok))
            self.write('Conexion comprobada.')
        self.background(work)

    def validate(self):
        def work():
            errors, used = check_module_graph.check('CopaNAOPenalty', 'CopaNAO_CCM2026')
            for error in errors:
                self.write('ERROR ' + error, 'error')
            self.write('Escenario CopaNAOPenalty: %d modulos, %d errores.' % (len(used), len(errors)),
                       'ok' if not errors else 'error')
        self.background(work)

    def build(self):
        config = self.config.get()
        self.background(lambda: self.write('Compilacion %s: %s' % (
            config, 'OK' if robot_deploy.build(config, self.printer('build')) == 0 else 'FALLO')))

    def options_for(self, row):
        start = next((k for k, v in START_LABELS.items() if v == self.start_mode.get()), 'buttonOnly')
        return robot_deploy.DeployOptions(
            role=row.role_name(), striker_policy=row.striker.get(), keeper_policy=row.keeper.get(),
            start_mode=start, player=self.safe_int(row.player, 1), team=self.safe_int(self.team, 0),
            jumps=self.jumps.get(), wifi_off=self.wifi_off.get(), volume=self.safe_int(self.volume, 70),
            restart=self.restart.get(), config=self.config.get(), dry_run=self.dry_run.get())

    def deploy(self):
        rows = self.selected()
        if not rows:
            return
        jobs = []
        try:
            for row in rows:
                options = self.options_for(row)
                robot_deploy.validate(options)
                jobs.append((row, options))
        except ValueError as error:
            messagebox.showerror('Configuracion invalida', str(error))
            return
        players = [options.player for _, options in jobs]
        if len(set(players)) != len(players) and not messagebox.askyesno(
                'Numeros repetidos', 'Hay numeros de jugador repetidos. Continuar?'):
            return
        if not self.wifi_off.get() and not messagebox.askyesno(
                'Wi-Fi encendido', 'El reglamento (2.7) prohibe redes inalambricas en competencia. Desplegar sin apagar el Wi-Fi?'):
            return
        summary = '\n'.join('%s: %s, tirador=%s, portero=%s, jugador %d' % (
            row.robot.name, ROLE_LABELS[o.role], o.striker_policy, o.keeper_policy, o.player) for row, o in jobs)
        if not self.dry_run.get() and not messagebox.askyesno('Confirmar despliegue', summary + '\n\nDesplegar ahora?'):
            return
        compile_first, config = self.compile.get(), self.config.get()

        def work():
            if compile_first:
                if robot_deploy.build(config, self.printer('build')) != 0:
                    self.write('La compilacion fallo; no se despliega.', 'error')
                    return
            elif robot_deploy.sync(self.printer('sync')) != 0:
                self.write('No se pudo preparar el arbol de compilacion.', 'error')
                return
            for row, _ in jobs:
                self.events.put(('result', row, 'desplegando...', 'busy'))

            def one(job):
                row, options = job
                code = robot_deploy.deploy(row.robot, options, self.printer(row.robot.name))
                self.events.put(('result', row, 'OK' if code == 0 else 'error %d' % code, 'ok' if code == 0 else 'error'))
                return code
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                codes = list(pool.map(one, jobs))
            done = sum(1 for code in codes if code == 0)
            self.write('Despliegue terminado: %d/%d robots%s.' % (done, len(codes), ' (simulacro)' if jobs[0][1].dry_run else ''),
                       'ok' if done == len(codes) else 'error')
        self.background(work)

    def on_selected(self, action):
        rows = self.selected()
        if not rows:
            return
        logs = ROOT / 'runs/robot_logs'

        def one(row):
            out = self.printer(row.robot.name)
            if action == 'status':
                return robot_deploy.status(row.robot, out)
            if action == 'logs':
                code = robot_deploy.download_logs(row.robot, logs, out)
                out('logs en %s' % (logs / row.robot.name))
                return code
            return robot_deploy.service(row.robot, action, out)

        def work():
            with ThreadPoolExecutor(max_workers=len(rows)) as pool:
                list(pool.map(one, rows))
        self.background(work)


def main():
    root = tk.Tk()
    DeployDialog(root)
    root.mainloop()


if __name__ == '__main__':
    main()
