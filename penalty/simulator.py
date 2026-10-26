"""Interactive penalties driven by the original B-Human movement engine."""
import argparse
import json
import math
import threading

from penalty.motion_contract import ROOT
from penalty.physical_sim import PhysicalSimulator
from penalty.sim_session import SimulationSession

PenaltySimulator = PhysicalSimulator


def run(headless=True, mode='penalty', seconds=28., action='stand', forward=.06,
        lateral=0., turn=0., kick_type='forwardFastLeft', block_side='center',
        role='striker', seed=0, record_dir=ROOT/'runs', ball_xy=(.1,0.)):
    if not math.isfinite(seconds) or not 0 < seconds <= 28:
        raise ValueError('seconds debe estar entre 0 y 28')
    sim = PhysicalSimulator(seed=seed, ball_xy=ball_xy)
    session = SimulationSession(sim, record_dir=record_dir, mode=mode)
    session.command('role', role)
    session.manual['kick_type'] = kick_type
    if mode == 'manual':
        session.command('manual', {'motion':action, 'forward':forward,
            'lateral':lateral, 'turn':turn, 'kick_type':kick_type,'block_side':block_side})
    if headless:
        try:
            session.command('start')
            for _ in range(int((seconds+8)/sim.dt)+1):
                session.tick()
                if session.state in ('FINISHED','ABORTED','ERROR','CLOSED'):
                    break
                if session.state == 'RUNNING' and session.penalty_time >= seconds-1e-9:
                    session._finish()
                    session._publish()
                    break
            else:
                raise RuntimeError('Se agotó el presupuesto del ensayo')
            print(json.dumps(session.snapshot(),allow_nan=False))
        finally:
            session.close()
            session.tick()
    else:
        import mujoco.viewer
        from penalty.sim_panel import run_panel
        session.viewer = mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=session.key)
        sim.mutation_lock = session.viewer.lock
        session.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        session.viewer.cam.fixedcamid = sim.model.camera('overview').id
        worker = threading.Thread(target=session.run,name='penalty-simulation',daemon=True)
        worker.start()
        try:
            run_panel(session)
        finally:
            session.close()
            worker.join(6)
            if worker.is_alive():
                raise RuntimeError('El trabajador de simulación no terminó en 6 segundos')
    return sim, [tuple(item['states']) for item in session.states]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--mode',choices=('penalty','manual'),default='penalty')
    parser.add_argument('--seconds',type=float,default=28.,help='duración activa sin ventana (máximo 28 s)')
    parser.add_argument('--action',choices=('stand','walk','kick','block','getup','sit'),default='stand')
    parser.add_argument('--role',choices=('striker','keeper'),default='striker')
    parser.add_argument('--forward',type=float,default=.06,help='m/s, modo manual')
    parser.add_argument('--lateral',type=float,default=0.,help='m/s, modo manual')
    parser.add_argument('--turn',type=float,default=0.,help='rad/s, modo manual')
    parser.add_argument('--kick-type',choices=('forwardFastLeft','forwardFastRight'),default='forwardFastLeft')
    parser.add_argument('--block-side',choices=('center','left','right'),default='center')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--ball-x',type=float,default=.1)
    parser.add_argument('--ball-y',type=float,default=0.)
    parser.add_argument('--record-dir',type=str,default=str(ROOT/'runs'))
    args = parser.parse_args()
    try:
        run(headless=args.headless,mode=args.mode,seconds=args.seconds,action=args.action,
            role=args.role,forward=args.forward,lateral=args.lateral,turn=args.turn,
            kick_type=args.kick_type,block_side=args.block_side,seed=args.seed,
            ball_xy=(args.ball_x,args.ball_y),record_dir=args.record_dir)
    except (RuntimeError,ValueError) as error:
        parser.exit(1,str(error)+'\n')


if __name__ == '__main__':
    main()
