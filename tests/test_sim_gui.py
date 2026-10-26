"""Actual Windows viewer plus Tk controls, using the genuine subprocesses."""
import importlib.util
import os
import threading
import time
import unittest
from unittest.mock import patch

tk = importlib.import_module('tkinter') if importlib.util.find_spec('tkinter') else None
if tk is not None:
    from penalty.sim_panel import ControlPanel
from penalty.simulator import run


@unittest.skipUnless(os.name=='nt','MSVC/Windows graphical integration')
@unittest.skipIf(tk is None,'tkinter no está instalado (sudo apt install python3-tk)')
class GraphicalIntegrationTests(unittest.TestCase):
    def test_viewer_panel_pause_step_reset_repeat_and_shutdown(self):
        failures = []
        observations = []

        def automated_panel(session):
            root = tk.Tk()
            root.withdraw()
            panel = ControlPanel(session,root)
            stage = 0
            paused_time = None
            previous_trial = None
            processes = []
            deadline = time.monotonic()+25

            def check():
                nonlocal stage,paused_time,previous_trial,processes
                try:
                    snapshot = session.snapshot()
                    self.assertLess(time.monotonic(),deadline,'GUI test timed out')
                    self.assertNotEqual(snapshot['state'],'ERROR',snapshot.get('error'))
                    if stage == 0:
                        self.assertEqual(snapshot['state'],'PREPARED')
                        self.assertEqual(snapshot['time_s'],0.)
                        panel.buttons['start'].invoke()
                        stage = 1
                    elif stage == 1 and snapshot['state']=='RUNNING':
                        panel.buttons['pause'].invoke()
                        stage = 2
                    elif stage == 2 and snapshot['state']=='PAUSED':
                        paused_time = snapshot['time_s']
                        previous_trial = snapshot['trial']
                        processes = [b.process for b in session.sim.bridges.values()]
                        stage = 3
                    elif stage == 3:
                        self.assertEqual(snapshot['time_s'],paused_time)
                        panel.buttons['step'].invoke()
                        stage = 4
                    elif stage == 4 and snapshot['time_s']>paused_time:
                        self.assertEqual(snapshot['state'],'PAUSED')
                        self.assertAlmostEqual(snapshot['time_s']-paused_time,.012)
                        panel.buttons['reset'].invoke()
                        stage = 5
                    elif stage == 5 and snapshot['state']=='PREPARED':
                        self.assertNotEqual(snapshot['trial'],previous_trial)
                        self.assertEqual(snapshot['time_s'],0.)
                        self.assertTrue(all(p.poll() is not None for p in processes))
                        panel.buttons['repeat'].invoke()
                        stage = 6
                    elif stage == 6 and snapshot['state']=='SETTLING':
                        observations.append('all controls and viewer boot verified')
                        panel.close()
                        return
                    root.after(150,check)
                except Exception as error:
                    failures.append(error)
                    panel.close()
            root.after(100,check)
            root.mainloop()

        with patch('penalty.sim_panel.run_panel',side_effect=automated_panel):
            sim,_ = run(headless=False,record_dir=None)
        if failures:
            raise failures[0]
        self.assertTrue(observations)
        self.assertEqual(sim.bridges,{})
        self.assertFalse(any(t.name=='penalty-simulation' and t.is_alive()
                             for t in threading.enumerate()))
