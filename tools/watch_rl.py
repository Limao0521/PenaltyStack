"""Watch or record penalties in B-Human: one striker against one keeper.

  python tools/watch_rl.py --striker models/rl/striker_v2 --keeper scripted --shots 4 --live
  python tools/watch_rl.py --striker models/rl/striker_v2 --keeper scripted --seeds 50000000 50000003 --video docs/media/rl

Players: a trained policy directory (':gate' for keepers trained with
commit_after_shot) or a baseline (scripted, geometric, center). Uses the
duel world (penalty.rl.duel) under a duel condition (default 'realista', the
training conditions), so a duel episode seed replays that same penalty.
--video writes one MP4 per shot; --live opens the MuJoCo viewer at real
speed. The camera is behind and above the striker: the left of the image is
the striker's left.
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from penalty.rl.duel import CONDITIONS, DuelWorld, duel_episode, make_player, player_spec  # noqa: E402
from penalty.rl.metrics import shot_side  # noqa: E402

TICKS_PER_FRAME = 3                     # 36 ms of simulation per video frame
SIDE_NAMES = {'left': 'izquierda', 'center': 'centro', 'right': 'derecha'}
SIDE_PHRASES = {'left': 'a la izquierda', 'center': 'al centro', 'right': 'a la derecha'}


def behind_camera(camera):
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-.78, 0., .2)
    camera.distance, camera.azimuth, camera.elevation = 2.9, 180., -26.
    return camera


class WatchedDuel(DuelWorld):
    """Calls on_tick(world) after every simulation tick (12 ms)."""

    def __init__(self, on_tick, **options):
        super().__init__(**options)
        self.on_tick = on_tick

    def _tick(self):
        super()._tick()
        self.on_tick(self)

    def linger(self, seconds):
        """Keep simulating after the result so the ending is visible."""
        if self.session is None:
            return
        end = self.session.penalty_time + seconds
        try:
            self._advance(until=lambda: self.session.penalty_time >= end)
        except RuntimeError:
            pass


class Recorder:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.renderer = self.camera = self.model = None
        self.frames, self.ticks = [], 0

    def __call__(self, world):
        if world.session.state != 'RUNNING':  # skip the 2 s settling before the penalty
            return
        self.ticks += 1
        if self.ticks % TICKS_PER_FRAME:
            return
        model = world.sim.model
        if self.renderer is None or self.model is not model:  # new session after a sim error
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
            self.renderer, self.model = mujoco.Renderer(model, self.height, self.width), model
            self.camera = behind_camera(mujoco.MjvCamera())
        self.renderer.update_scene(world.sim.data, camera=self.camera)
        self.frames.append(self.renderer.render())

    def save(self, path):
        import imageio.v2 as imageio
        imageio.mimwrite(path, self.frames, fps=1/(TICKS_PER_FRAME*.012), quality=7,
                         macro_block_size=2)
        self.frames, self.ticks = [], 0


class LiveViewer:
    """MuJoCo viewer synced every tick and paced to real time."""

    def __init__(self):
        self.viewer = None
        self.wall = self.sim_time = None

    def __call__(self, world):
        sim = world.sim
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            behind_camera(self.viewer.cam)
        if self.wall is None or sim.data.time < self.sim_time:
            self.wall, self.sim_time = time.monotonic(), sim.data.time
        ahead = (sim.data.time - self.sim_time) - (time.monotonic() - self.wall)
        if ahead > 0:
            time.sleep(ahead)
        self.viewer.sync()

    def running(self):
        return self.viewer is None or self.viewer.is_running()


def describe(record):
    side = None if record.get('target_y') is None else shot_side(record['target_y'])
    crossing = record.get('goal_line_y')
    return '%s, apunta %s (%+.2f m), cruza en %s' % (
        record['result'], SIDE_PHRASES.get(side, '-'), record.get('target_y') or 0.,
        '-' if crossing is None else '%+.2f m' % crossing)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--striker', required=True)
    parser.add_argument('--keeper', default='scripted')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='semillas de episodio del duelo')
    parser.add_argument('--shots', type=int, default=4, help='con --seeds ausente: 50000000, 50000001, ...')
    parser.add_argument('--condition', choices=sorted(CONDITIONS), default='realista')
    parser.add_argument('--video', type=Path, default=None, help='carpeta para un MP4 por tiro')
    parser.add_argument('--live', action='store_true', help='visor de MuJoCo a velocidad real')
    parser.add_argument('--size', type=int, nargs=2, default=(960, 540), metavar=('ANCHO', 'ALTO'))
    parser.add_argument('--linger', type=float, default=1.5, help='segundos extra tras el resultado')
    args = parser.parse_args()
    if not args.video and not args.live:
        parser.error('elige --video y/o --live')
    (striker_name, striker_spec), (keeper_name, keeper_spec) = player_spec(args.striker), player_spec(args.keeper)
    striker, keeper = make_player(striker_spec, 'striker'), make_player(keeper_spec, 'keeper')
    seeds = args.seeds or [50_000_000 + i for i in range(args.shots)]
    setting = CONDITIONS[args.condition]
    recorder = Recorder(*args.size) if args.video else None
    live = LiveViewer() if args.live else None

    def on_tick(world):
        if recorder:
            recorder(world)
        if live:
            live(world)

    world = WatchedDuel(on_tick, keeper_gate=keeper_spec.get('gate', False), seed=50, **setting['world'])
    if args.video:
        args.video.mkdir(parents=True, exist_ok=True)
    try:
        for number, seed in enumerate(seeds, 1):
            record = duel_episode(world, striker, keeper, seed, setting['noise'])
            world.linger(args.linger)
            print('tiro %d (semilla %d): %s' % (number, seed, describe(record)), flush=True)
            if recorder:
                side = SIDE_NAMES.get(None if record.get('target_y') is None else shot_side(record['target_y']), 'x')
                path = args.video/('tiro%d_%s_vs_%s_%s_%s.mp4' % (number, striker_name, keeper_name, side,
                                                                 record['result']))
                recorder.save(path)
                print('   video: %s' % path, flush=True)
            if live and not live.running():
                break
    finally:
        world.close()


if __name__ == '__main__':
    main()
