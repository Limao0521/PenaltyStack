"""Build offline RGB body-yaw references; never modify an active penalty.

Labels are known static reference-scene rotations. Reference articulated
poses can be collected from B-Human simulation; no pose labels enter runtime.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from penalty.nao_model import load_model, set_initial_pose
from penalty.opponent_orientation import VERSION, body_feature


def build(output, poses=None, seed=67001):
    rng = np.random.default_rng(seed)
    model = load_model(mocap=False)
    data = mujoco.MjData(model)
    set_initial_pose(model, data)
    base = data.qpos.copy()
    root = model.joint('striker_root_free').qposadr[0]
    own = model.joint('keeper_root_free').qposadr[0]
    # This is an isolated photographic reference scene. No integration,
    # root trajectories, ball velocities or scored penalties are generated.
    base[own:own+7] = [-1.08, 0, .334, 1, 0, 0, 0]
    base[model.joint('keeper_HeadPitch').qposadr[0]] = -.55
    poses = poses or [{}]
    features, yaws = [], []
    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        for pose_index, pose in enumerate(poses):
            for angle_deg in range(-180, 180, 10):
                for distance in (1.2, 1.65, 2.1):
                    data.qpos[:] = base
                    angle = math.radians(angle_deg)
                    lateral = float(rng.uniform(-.15, .15))
                    # Euler yaw-pitch-roll references; yaw labels specify the
                    # torso's forward axis, independently of feet/head turns.
                    cr, sr = math.cos(pose.get('torso_roll', 0.)/2), math.sin(pose.get('torso_roll', 0.)/2)
                    cp, sp = math.cos(pose.get('torso_pitch', 0.)/2), math.sin(pose.get('torso_pitch', 0.)/2)
                    cy, sy = math.cos(angle/2), math.sin(angle/2)
                    quat = [cy*cp*cr+sy*sp*sr, cy*cp*sr-sy*sp*cr,
                            cy*sp*cr+sy*cp*sr, sy*cp*cr-cy*sp*sr]
                    data.qpos[root:root+7] = [distance-1.08, lateral, .334, *quat]
                    for name, value in pose.get('joints', pose).items():
                        data.qpos[model.joint('striker_'+name).qposadr[0]] = value
                    mujoco.mj_forward(model, data)
                    # Place the offline articulated reference on its soles.
                    feet = [model.body('striker_'+s+'Foot').id for s in ('L', 'R')]
                    data.qpos[root+2] -= min(data.xpos[f, 2] for f in feet)-.04511
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera='keeper_camera_lower')
                    rgb = renderer.render()
                    feature, _ = body_feature(rgb, (0, 0, 319, 239))
                    if feature is not None:
                        features.append(np.rint(feature*255).astype(np.uint8))
                        camera = model.camera('keeper_camera_lower').id
                        ray_yaw = math.atan2(lateral-data.cam_xpos[camera, 1],
                                            distance-1.08-data.cam_xpos[camera, 0])
                        yaws.append((angle-ray_yaw+math.pi) % (2*math.pi)-math.pi)
            print('Referencias de pose %d/%d: %d' % (pose_index+1, len(poses), len(features)), flush=True)
    finally:
        renderer.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output/'templates.npz'
    np.savez_compressed(target, features=np.asarray(features, dtype=np.uint8), yaws=np.asarray(yaws))
    source = ('models/bhuman/NaoV6H25.rsi3', 'models/bhuman/nao.xml', 'sim/scene.xml',
              'penalty/opponent_orientation.py', 'tools/build_orientation_templates.py',
              'tools/collect_orientation_poses.py', 'baseline/motion_manifest.json')
    metadata = {'version': VERSION, 'seed': seed, 'samples': len(yaws), 'angle_step_deg': 10,
                'original_bhuman_commit': '2360ab1100ad28f70e190c893f6b095651ff0bbc',
                'poses': poses, 'reference_labels': 'offline static articulated NAO RGB renders only',
                'runtime_inputs': 'RGB crop, detected robot box, calibrated measured bearing and own rotation',
                'domain': 'MuJoCo NAO V6 appearance only; not calibrated for physical NAO',
                'hardware_validated': False, 'max_match_distance': .025, 'max_angular_sigma_rad': .4,
                'templates_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source}}
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'models/vision/nao_yaw_v1')
    parser.add_argument('--poses', type=Path, help='offline original-controller joint pose JSON list')
    parser.add_argument('--seed', type=int, default=67001)
    args = parser.parse_args()
    pose_data = json.loads(args.poses.read_text()) if args.poses else None
    poses = pose_data['poses'] if isinstance(pose_data, dict) else pose_data
    print(json.dumps(build(args.output, poses, args.seed), indent=2))
