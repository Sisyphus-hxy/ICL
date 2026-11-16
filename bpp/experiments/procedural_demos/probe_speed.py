"""Calibrate the bounded-rate servo: measure actual eef speed for several
waypoint-count / steps-per-waypoint combos on demo_0's transport path."""
import argparse, os, sys
parser = argparse.ArgumentParser()
parser.add_argument('--egl-device', default='1')
a = parser.parse_args()
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', a.egl_device)
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ.setdefault('PYNPUT_BACKEND', 'dummy')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'clearance'))
sys.path.insert(0, str(HERE))

import numpy as np
import h5py
from scipy.spatial.transform import Rotation, Slerp
from scene_lib import SemanticScene, SOURCE
from expert import Servo, resample

scene = SemanticScene(record_images=False, water=False)
with h5py.File(SOURCE) as f:
    d = f['data/demo_0']
    info = dict(state=d['states'][0].copy(), actions=d['actions'][:],
                ee_pos=d['obs/ee_pos'][:], ee_ori=d['obs/ee_ori'][:])
scene.canonical_state = info['state']
T0, T1 = 75, 93
DT = 0.05

for n_wp, steps_per in [(45, 3), (15, 2), (10, 2), (8, 2), (6, 2), (4, 2)]:
    scene.reset()
    servo = Servo(scene.env)
    for i in range(T0):
        scene.env.step(info['actions'][i])
    path = resample(info['ee_pos'][T0 - 1:T1], n_wp)
    q0 = Rotation.from_rotvec(info['ee_ori'][T0 - 1])
    q1 = Rotation.from_rotvec(info['ee_ori'][T1 - 1])
    sl = Slerp([0., 1.], Rotation.from_quat([q0.as_quat(), q1.as_quat()]))
    traj = []
    orig_step = scene.env.step

    def counting(action):
        r = orig_step(action)
        traj.append(r[0]['robot0_eef_pos'].copy())
        return r

    scene.env.step = counting
    for k, p in enumerate(path):
        target = np.r_[p, sl((k + 1) / n_wp).as_quat()]
        servo.move_to_pose(target, close_gripper=True, max_steps=steps_per,
                           pos_threshold=0., rot_threshold=0.,
                           max_pos_speed=0.15, max_rot_speed=0.3)
    scene.env.step = orig_step
    traj = np.asarray(traj)
    v = np.linalg.norm(np.diff(traj, axis=0), axis=1) / DT
    L = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()
    print(f'n_wp={n_wp:3d} steps_per={steps_per} -> {len(traj):3d} steps, '
          f'L={L * 1000:.0f}mm, v_mean={v.mean():.4f} v_p95={np.percentile(v, 95):.4f} m/s',
          flush=True)
scene.close()
print('probe finished')
