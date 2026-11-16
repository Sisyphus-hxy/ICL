"""The installed LIBERO bowl-on-cookies scene, with one relocatable MuJoCo post."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner

BPP_ROOT = Path(__file__).resolve().parents[2]
TASK = 'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box'
SPLIT = 'libero_spatial_selected_combinations_view'
OBJECT = 'akita_black_bowl_1'
GOAL = 'cookies_1'
OBSTACLE_RADIUS = 0.025
OBSTACLE_HEIGHT = 0.18
LAYOUTS = {'original': np.array([-0.020, 0.025, 0.99]),
           'relocated': np.array([-0.020, 0.145, 0.99])}


def state_hash(state):
    return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()


class Scene:
    def __init__(self, resolution=128, record_images=True):
        bddl = BPP_ROOT / 'src/behavior_prompting/train_network/env/libero/bddl_files' / SPLIT / (TASK + '.bddl')
        self.source = BPP_ROOT / 'datasets/libero' / SPLIT / (TASK + '_demo.hdf5')
        if not self.source.is_file():
            raise FileNotFoundError(f'Use the already prepared smoke demonstration: {self.source}')
        with h5py.File(self.source, 'r') as f:
            demo = f['data/demo_0']
            self.initial_state = demo['states'][0].copy()
            self.reference_actions = demo['actions'][:]
            self.reference_seed = int(demo.attrs.get('seed', 2))
            self.reference_grasp_pose = np.r_[demo['obs/ee_pos'][55],
                Rotation.from_rotvec(demo['obs/ee_ori'][55]).as_quat()]
        self.wrapper = ControlEnvForRunner(str(bddl), camera_heights=resolution,
                                          camera_widths=resolution, control_delta=True,
                                          use_camera_obs=record_images,
                                          has_offscreen_renderer=record_images,
                                          ignore_done=True, hard_reset=False)
        self.env = self.wrapper.env
        self.env.seed(self.reference_seed)
        self.env.reset()
        # Reuse the complete installed scene and add a single mocap body. It can
        # be relocated between episodes, but remains stationary during each one.
        tree = ET.fromstring(self.env.sim.model.get_xml())
        body = ET.SubElement(tree.find('worldbody'), 'body', name='procedure_obstacle',
                             mocap='true', pos='-0.020 0.025 0.99')
        ET.SubElement(body, 'geom', name='procedure_obstacle_geom', type='cylinder',
                      size=f'{OBSTACLE_RADIUS} {OBSTACLE_HEIGHT / 2}',
                      group='1',
                      rgba='0.85 0.23 0.08 1', friction='0.8 0.01 0.001',
                      contype='1', conaffinity='1')
        self.env.reset_from_xml_string(ET.tostring(tree, encoding='unicode'))
        self.env.hard_reset = False
        self.model_xml = self.env.sim.model.get_xml()
        self.m = self.env.sim.model._model
        self.d = self.env.sim.data._data
        self.obstacle_id = self.env.sim.model.geom_name2id('procedure_obstacle_geom')
        body_id = self.env.sim.model.body_name2id('procedure_obstacle')
        self.mocap_id = int(self.m.body_mocapid[body_id])
        self.object_body = self.env.obj_body_id[OBJECT]
        self.goal_body = self.env.obj_body_id[GOAL]
        self.object_geoms = self.geoms_for_bodies([self.object_body])
        self.robot_geoms = [i for i in range(self.m.ngeom)
                            if self.m.geom_bodyid[i] > 0 and self.geom_name(i).startswith(('robot0_', 'gripper0_'))
                            and self.collidable(i)]
        self.gripper_geoms = [i for i in self.robot_geoms if self.geom_name(i).startswith('gripper0_')]
        self.record_images = record_images
        self.reset('original')
        self.canonical_state = self.env.sim.get_state().flatten().copy()

    def collidable(self, i):
        return bool(self.m.geom_contype[i] or self.m.geom_conaffinity[i])

    def geom_name(self, i):
        return self.env.sim.model.geom_id2name(i) or ''

    def geoms_for_bodies(self, ids):
        return [i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] in ids and self.collidable(i)]

    def reset(self, layout):
        np.random.seed(self.reference_seed)
        self.env.seed(self.reference_seed)
        self.env.reset()
        state = getattr(self, 'canonical_state', self.initial_state)
        self.env.sim.set_state_from_flattened(state)
        if not hasattr(self, 'canonical_state'):
            self.env.sim.forward()
            self.reference_object_pos = self.object_pos()
            # Increase the transport distance while preserving this LIBERO task's
            # objects, goal predicate, robot, and grasp. Only the goal box moves.
            joint = self.env.objects_dict[GOAL].joints[0]
            q = self.env.sim.data.get_joint_qpos(joint).copy()
            q[:2] = [0.130, 0.025]
            self.env.sim.data.set_joint_qpos(joint, q)
            joint = self.env.objects_dict[OBJECT].joints[0]
            q = self.env.sim.data.get_joint_qpos(joint).copy()
            q[0] = -0.170
            self.env.sim.data.set_joint_qpos(joint, q)
        self.d.mocap_pos[self.mocap_id] = LAYOUTS[layout]
        self.d.mocap_quat[self.mocap_id] = [1, 0, 0, 0]
        self.env.sim.forward()
        self.env.robots[0].controller.update(force=True)
        self.env.robots[0].controller.reset_goal()
        gripper=self.env.robots[0].gripper
        gripper.current_action = (self.restore_gripper_action.copy()
            if hasattr(self, 'restore_gripper_action') else np.zeros(2))
        self.env._update_observables(force=True)
        self.layout = layout
        return self.obs()

    def obs(self):
        obs = self.env._get_observations(force_update=True)
        return obs

    def object_pos(self):
        return self.d.xpos[self.object_body].copy()

    def goal_pos(self):
        return self.d.xpos[self.goal_body].copy()

    def initial_identity(self):
        state = self.env.sim.get_state().flatten()
        return {'sim_state_sha256': state_hash(state),
                'mocap_pos': self.d.mocap_pos.copy().tolist(),
                'gripper_command_state':self.env.robots[0].gripper.current_action.tolist(),
                'controller_goal_pos':self.env.robots[0].controller.goal_pos.tolist(),
                'controller_goal_ori':self.env.robots[0].controller.goal_ori.tolist(),
                'obstacle_pos': LAYOUTS[self.layout].tolist()}

    def surface_clearance(self, geoms):
        out = np.empty(6)
        return min(float(mujoco.mj_geomDistance(self.m, self.d, i, self.obstacle_id, 2., out))
                   for i in geoms)

    def contacts(self, transport=False):
        robot = set(self.robot_geoms)
        obj = set(self.object_geoms)
        gripper = set(self.gripper_geoms)
        collisions = []
        for c in self.d.contact[:self.d.ncon]:
            if c.dist > 0:
                continue
            a, b = int(c.geom1), int(c.geom2)
            pair = {a, b}
            # Bowl/gripper contact is the intended grasp. Robot self-contacts
            # with disabled collision pairs do not appear in MuJoCo's contacts.
            if (a in gripper and b in obj) or (b in gripper and a in obj):
                continue
            if self.obstacle_id in pair and pair & (robot | obj):
                collisions.append((self.geom_name(a), self.geom_name(b), float(c.dist)))
            elif a in robot and b in robot and not (a in gripper and b in gripper):
                collisions.append((self.geom_name(a), self.geom_name(b), float(c.dist)))
            elif (a in robot) != (b in robot):
                # Fixed robot base / floor contact is structural support.
                other = b if a in robot else a
                if self.geom_name(other) not in ('floor',):
                    collisions.append((self.geom_name(a), self.geom_name(b), float(c.dist)))
            elif transport and ((a in obj) != (b in obj)) and not pair & gripper:
                collisions.append((self.geom_name(a), self.geom_name(b), float(c.dist)))
        return collisions

    def close(self):
        self.wrapper.close()
