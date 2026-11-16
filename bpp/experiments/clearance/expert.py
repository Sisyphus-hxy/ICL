"""Reuse the upstream proportional OSC controller; alter transport waypoints only."""
import sys

import numpy as np
from scipy.optimize import brentq

from scene import BPP_ROOT, LAYOUTS

# Reuse the upstream generator's pose servo itself, without invoking its global
# task enumeration or requiring reference datasets that have not been downloaded.
sys.path.insert(0, str(BPP_ROOT / 'src/behavior_prompting/train_network/scripts/libero'))
from generate_demonstrations import TaskDemonstrationGenerator


class Servo:
    move_to_pose = TaskDemonstrationGenerator.move_to_pose

    def __init__(self, env):
        self.env = env
        self.base_split_name = 'libero_spatial'
        self.max_height = {'libero_spatial': 1.3}
        self.rot_gain_position_scaling = None
        self.log_position_error_on_fail = False

    def _print(self, *args, **kwargs):
        pass


def resample(points, n):
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return np.stack([np.interp(np.linspace(0, distances[-1], n), distances, points[:, k])
                     for k in range(3)], axis=-1)


def path_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def transport_paths(start, end, layout):
    t = np.linspace(0, 1, 2001)
    # The original scene uses positive-y detours. Relocation blocks those paths;
    # the independent feasibility witnesses pass on the negative-y side instead.
    peak_narrow, peak_wide = ((0.185, 0.250) if layout == 'original' else (0.090, 0.015))

    def curve(peak, compensate=0.):
        p = start[None] * (1-t[:, None]) + end[None] * t[:, None]
        midpoint_y = (start[1] + end[1]) / 2
        p[:, 1] += (peak - midpoint_y) * np.sin(np.pi*t)**2
        # A smooth extra excursion far before the post matches path length.
        # Arc-length resampling then matches duration and commanded speed too.
        u = np.clip(t / .40, 0, 1)
        p[:, 0] -= compensate * np.sin(np.pi*u)**2
        return p

    wide = curve(peak_wide)
    target = path_length(wide)
    amount = brentq(lambda x: path_length(curve(peak_narrow, x))-target, 0., .30)
    narrow = curve(peak_narrow, amount)
    return {'narrow': resample(narrow, 150), 'wide': resample(wide, 150)}


def generate(scene, recorder, procedure, layout):
    scene.reset(layout)
    recorder.start(scene)
    servo = Servo(recorder)
    recorder.phase = 'grasp'
    # As in the upstream generator, transfer the successful local demonstration's
    # grasp pose relative to the bowl. Translation enlarges the transport corridor.
    grasp_pose = scene.reference_grasp_pose.copy()
    grasp_pose[:3] += scene.object_pos()-scene.reference_object_pos
    def fixed_move(pose, steps, closed=True):
        for _ in range(steps // 2):
            servo.move_to_pose(pose, close_gripper=closed, max_steps=2,
                               pos_threshold=0., rot_threshold=0.)
    above = grasp_pose.copy(); above[2] += .10
    fixed_move(above, 80, closed=False)
    fixed_move(grasp_pose, 80, closed=False)
    for _ in range(20):
        recorder.step(np.array([0., 0., 0., 0., 0., 0., 1.]))
    recorder.phase = 'lift'
    start = grasp_pose[:3].copy()
    start[2] = 1.075
    lift = np.r_[start, grasp_pose[3:]]
    fixed_move(lift, 110)
    assert np.linalg.norm(scene.obs()['robot0_eef_pos']-start)<.006, 'Lift failed'
    assert scene.object_pos()[2]>.97, 'Reference grasp did not lift the bowl'
    # Both procedures receive exactly the same grasp, lift, orientation and place
    # targets. Only this middle segment is changed.
    end = start.copy()
    end[:2] += scene.goal_pos()[:2] - scene.object_pos()[:2]
    place = end.copy()
    place[2] = scene.goal_pos()[2] + .010 + (start[2]-scene.object_pos()[2])
    paths = transport_paths(start, end, layout)
    recorder.phase = 'transport'
    for point in paths[procedure]:
        servo.move_to_pose(np.r_[point, grasp_pose[3:]], close_gripper=True,
                           max_steps=2, pos_threshold=0., rot_threshold=0.)
    recorder.phase = 'align'
    fixed_move(np.r_[end, grasp_pose[3:]], 50)
    assert np.linalg.norm(scene.obs()['robot0_eef_pos']-end)<.006, 'Alignment failed'
    # Use the common grasp-relative vertical offset to lower onto the box.
    recorder.phase = 'place'
    fixed_move(np.r_[place, grasp_pose[3:]], 100)
    recorder.phase = 'release'
    for _ in range(16):
        recorder.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
    recorder.finish(scene)
    return {'grasp_target': grasp_pose.tolist(), 'transport_start': start.tolist(),
            'transport_end': end.tolist(), 'place_target': place.tolist(),
            'commanded_transport_length_m': path_length(paths[procedure])}
