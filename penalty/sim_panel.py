"""Tk control panel; all simulation work is queued to its owning thread."""
import tkinter as tk
from tkinter import ttk
from penalty.sim_session import BLOCK_SIDES


class ControlPanel:
    def __init__(self, session, root=None):
        self.session = session
        self.root = root or tk.Tk()
        self.root.title('PenaltyStack · Control de simulación')
        self.root.geometry('670x600')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        frame = ttk.Frame(self.root,padding=12)
        frame.pack(fill='both',expand=True)
        self.status = tk.StringVar()
        self.detail = tk.StringVar()
        self.error = tk.StringVar()
        ttk.Label(frame,textvariable=self.status,font=('Segoe UI',12,'bold')).pack(anchor='w')
        ttk.Label(frame,textvariable=self.detail,wraplength=630).pack(anchor='w',pady=8)
        row = ttk.Frame(frame)
        row.pack(fill='x')
        self.buttons = {}
        for action,label in [('start','Iniciar'),('pause','Pausa / Continuar'),
                             ('reset','Reiniciar'),('repeat','Repetir'),('step','Paso 12 ms'),('abort','ABORT')]:
            button = ttk.Button(row,text=label,command=lambda a=action:self.session.enqueue(a))
            button.pack(side='left',padx=2)
            self.buttons[action] = button
        settings = ttk.Frame(frame)
        settings.pack(fill='x',pady=12)
        self.mode = tk.StringVar(value=session.snapshot()['mode'])
        self.role = tk.StringVar(value='striker')
        self.speed = tk.StringVar(value='1')
        for label,variable,values,action in [('Modo',self.mode,['penalty','manual'],'mode'),
                ('Robot',self.role,['striker','keeper'],'role'),
                ('Velocidad',self.speed,['0.25','0.5','1','2'],'speed')]:
            ttk.Label(settings,text=label).pack(side='left',padx=4)
            box = ttk.Combobox(settings,textvariable=variable,values=values,state='readonly',width=10)
            box.pack(side='left')
            box.bind('<<ComboboxSelected>>',lambda e,a=action,v=variable:self.session.enqueue(a,v.get()))
        manual = ttk.LabelFrame(frame,text='Acciones originales B-Human · modo manual',padding=8)
        manual.pack(fill='x')
        self.fields = {}
        defaults = [('forward','Avance (m/s)',0.06),('lateral','Lateral (m/s)',0.),
                    ('turn','Giro (rad/s)',0.),('head_yaw','Cabeza yaw (rad)',0.),
                    ('head_pitch','Cabeza pitch (rad)',0.)]
        for i,(name,label,default) in enumerate(defaults):
            ttk.Label(manual,text=label).grid(row=i,column=0,sticky='w')
            variable = tk.StringVar(value=str(default))
            ttk.Entry(manual,textvariable=variable,width=12).grid(row=i,column=1,padx=8,pady=2)
            self.fields[name] = variable
        self.kick_type = tk.StringVar(value='forwardFastLeft')
        self.block_side = tk.StringVar(value='center')
        for i,(label,variable,values) in enumerate([
                ('Patada',self.kick_type,['forwardFastLeft','forwardFastRight']),
                ('Bloqueo',self.block_side,list(BLOCK_SIDES))],start=5):
            ttk.Label(manual,text=label).grid(row=i,column=0,sticky='w')
            ttk.Combobox(manual,textvariable=variable,values=values,state='readonly',width=22).grid(row=i,column=1,sticky='w')
        actions = ttk.Frame(manual)
        actions.grid(row=7,column=0,columnspan=2,pady=8)
        for motion,label in [('stand','Pararse'),('walk','Caminar'),('kick','Patear'),
                             ('block','Bloquear'),('sit','Sentarse'),('getup','Levantarse')]:
            ttk.Button(actions,text=label,command=lambda m=motion:self.request_motion(m)).pack(side='left',padx=2)
        ttk.Label(frame,text='Espacio: iniciar/pausa · R: reiniciar · Enter: repetir · .: paso · A: abortar\n'
                  'Cambiar modo reinicia el ensayo. Reiniciar deja la escena pausada.\n'
                  'Límites manuales: avance/lateral ±0.15 m/s; giro ±0.35 rad/s.').pack(anchor='w',pady=8)
        ttk.Label(frame,textvariable=self.error,foreground='#b02020',wraplength=630).pack(anchor='w')
        # Shortcuts in the viewer are always active. In this panel, ignore
        # shortcuts while editing numbers so decimal points remain usable.
        self.root.bind('<KeyPress>',self.key)
        self.poll()

    def key(self, event):
        if isinstance(event.widget,(ttk.Entry,ttk.Combobox,tk.Entry)):
            return
        keys = {'space':32,'r':82,'R':82,'Return':257,'period':46,'a':65,'A':65}
        if event.keysym in keys:
            self.session.key(keys[event.keysym])

    def request_motion(self, motion):
        try:
            request = {name:float(variable.get()) for name,variable in self.fields.items()}
            request.update(motion=motion,kick_type=self.kick_type.get(),block_side=self.block_side.get())
            # Validate here too: an editing error must not stop the worker.
            from penalty.sim_session import SimulationSession
            SimulationSession._validate_manual(request)
            if self.mode.get() != 'manual':
                raise ValueError('Selecciona modo manual para ejecutar esta acción.')
            self.session.enqueue('manual',request)
            self.error.set('')
        except ValueError as error:
            self.error.set(str(error))

    def poll(self):
        snapshot = self.session.snapshot()
        if snapshot['state'] == 'CLOSED':
            self.root.destroy()
            return
        self.status.set('%s · ensayo %s · %.2f / 28 s · %.2fx' % (
            snapshot['state'],(snapshot.get('trial') or 'nuevo')[:8],
            snapshot['penalty_time_s'],snapshot['speed']))
        robots = snapshot['robots']
        detail = [snapshot['message'],'Resultado: '+snapshot['result']]
        for role,label in [('striker','Tirador'),('keeper','Portero')]:
            robot = robots[role]
            detail.append('%s: %s / %s · caída: %s · carga pies: %.2f kg' % (
                label,robot['state'],robot['phase'],robot['fallen'],sum(robot['fsr'])))
        self.detail.set('\n'.join(detail))
        if snapshot.get('error'):
            self.error.set(snapshot['error'])
        state = snapshot['state']
        self.buttons['start'].configure(state='normal' if state in ('PREPARED','FINISHED','ABORTED') else 'disabled')
        self.buttons['pause'].configure(state='normal' if state in ('RUNNING','SETTLING','PAUSED') else 'disabled')
        self.buttons['step'].configure(state='normal' if state in ('PREPARED','PAUSED') else 'disabled')
        self.root.after(50,self.poll)

    def close(self):
        self.session.close()
        self.root.destroy()


def run_panel(session):
    panel = ControlPanel(session)
    panel.root.mainloop()
