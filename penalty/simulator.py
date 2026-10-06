"""Flujo de penales con el modelo NAO V6 B-Human y raíces cinemáticas."""
import argparse
import math
import mujoco

from penalty.control import Observation, PenaltyController
from penalty.nao_model import KINEMATIC_ROOT_Z, load_model, set_initial_pose


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PenaltySimulator:
    dt = 0.1

    def __init__(self):
        self.model = load_model(mocap=True)
        self.data = mujoco.MjData(self.model)
        self.striker_mocap = self.model.body('striker_Nao').mocapid[0]
        self.keeper_mocap = self.model.body('keeper_Nao').mocapid[0]
        joint = self.model.joint('ball_free')
        self.ball_qpos = joint.qposadr[0]
        self.ball_qvel = joint.dofadr[0]
        self.striker = [0.3, 0.0, math.pi]
        self.keeper = [-1.08, 0.0, 0.0]
        self.kick_count = 0
        self.min_ball_x = 0.1
        self.reset()

    def _mocap(self, mocap_id, pose):
        self.data.mocap_pos[mocap_id] = [pose[0], pose[1], KINEMATIC_ROOT_Z]
        yaw = pose[2]
        self.data.mocap_quat[mocap_id] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        set_initial_pose(self.model, self.data)
        self.striker[:] = [0.3, 0.0, math.pi]
        self.keeper[:] = [-1.08, 0.0, 0.0]
        self.kick_count = 0
        self.min_ball_x = 0.1
        self._mocap(self.striker_mocap, self.striker)
        self._mocap(self.keeper_mocap, self.keeper)
        self.data.qpos[self.ball_qpos:self.ball_qpos + 7] = [0.1, 0, 0.0325, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self):
        bx, by = self.data.qpos[self.ball_qpos:self.ball_qpos + 2]
        result = {}
        for role, pose, goal in (
            ('striker', self.striker, (-1.2, 0.0)),
            ('keeper', self.keeper, (0.1, 0.0)),
        ):
            dx, dy = bx - pose[0], by - pose[1]
            gd_x, gd_y = goal[0] - pose[0], goal[1] - pose[1]
            result[role] = {
                'ball_range': math.hypot(dx, dy),
                'ball_bearing': wrap(math.atan2(dy, dx) - pose[2]),
                'goal_bearing': wrap(math.atan2(gd_y, gd_x) - pose[2]),
            }
        return result

    def step(self, striker_vx=0.0, striker_vy=0.0, striker_yaw_rate=0.0,
             keeper_vy=0.0, kick=False):
        """Avanza 0,1 s. Velocidades en marco del robot; kick es impulso ideal."""
        sx, sy, yaw = self.striker
        sx += (math.cos(yaw) * striker_vx - math.sin(yaw) * striker_vy) * self.dt
        sy += (math.sin(yaw) * striker_vx + math.cos(yaw) * striker_vy) * self.dt
        self.striker[:] = [min(1.15, max(-0.7, sx)), min(1.0, max(-1.0, sy)),
                           wrap(yaw + striker_yaw_rate * self.dt)]
        self.keeper[1] = min(0.6, max(-0.6, self.keeper[1] + keeper_vy * self.dt))
        self._mocap(self.striker_mocap, self.striker)
        self._mocap(self.keeper_mocap, self.keeper)
        if kick and self.kick_count == 0:
            bx, by = self.data.qpos[self.ball_qpos:self.ball_qpos + 2]
            if math.hypot(bx - self.striker[0], by - self.striker[1]) <= 0.25:
                direction = self.striker[2]
                self.data.qvel[self.ball_qvel] = math.cos(direction) * 2.7
                self.data.qvel[self.ball_qvel + 1] = math.sin(direction) * 2.7
                self.kick_count = 1
        for _ in range(int(self.dt / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self.min_ball_x = min(self.min_ball_x, self.data.qpos[self.ball_qpos])
        return self.observation()

    def result(self):
        bx, by = self.data.qpos[self.ball_qpos:self.ball_qpos + 2]
        if bx < -1.2 and abs(by) < 0.75:
            return 'goal'
        if bx < -1.2:
            return 'wide'
        if abs(by) > 1.2 or bx > 1.2:
            return 'out'
        if self.kick_count and self.min_ball_x < -0.9 and bx > self.min_ball_x + 0.1:
            return 'blocked'
        return 'in_play'


def run(headless=True):
    sim = PenaltySimulator()
    striker = PenaltyController('striker')
    keeper = PenaltyController('keeper')
    viewer = None
    if not headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    states = []
    for tick in range(300):
        t = tick * sim.dt
        values = sim.observation()
        s = striker.update(Observation(t, start=t >= 0.3, **values['striker']))
        k = keeper.update(Observation(t, start=t >= 0.3, **values['keeper']))
        if not states or states[-1] != (s.state, k.state):
            states.append((s.state, k.state))
            print('t=%.1f striker=%s keeper=%s' % (t, s.state, k.state))
        sim.step(s.forward, s.lateral, s.turn, k.lateral, s.kick)
        if viewer:
            viewer.sync()
        if s.state == 'STOP' and k.state == 'STOP':
            break
    if viewer:
        viewer.close()
    print('result=%s kick_count=%d ball=(%.3f, %.3f)' %
          (sim.result(), sim.kick_count, sim.data.qpos[sim.ball_qpos], sim.data.qpos[sim.ball_qpos + 1]))
    return sim, states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='run without window')
    args = parser.parse_args()
    run(headless=args.headless)
