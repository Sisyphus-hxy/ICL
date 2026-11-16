"""Native LIBERO pick-and-place scene with optional semantic attachments.

- ``water=True``: a light-blue mocap disk that tracks the bowl's XY position
  and stays level, rendering a liquid surface inside the carried vessel.
  It is a separate mocap body: the bowl's collision geoms, mass and control
  interface are untouched. Purely visual.
- ``n_drops>0``: free blue spheres placed inside the bowl — REAL dynamics.
  They rest inside the level bowl and spill out when the vessel tilts far
  enough. This makes "filled" a physical condition, not only a visual one:
  an Empty scene parks them outside the workspace. Spilling is detected by
  ``spill_count`` (drop beyond the bowl footprint or below its rim level).
- ``post_rgba`` given: the clearance-experiment mocap post near the transport
  corridor, with a parameterizable material (opaque plastic vs translucent
  glass). Position, size and collision geometry are identical across
  materials; only appearance differs.

The class deliberately does NOT implement the C0 baseline's canonical-state
relocation: ``reset`` always restores the state supplied via
``canonical_state`` (or the source demo_0 initial state) so generated data
uses native LIBERO geometry.

Free-joint drop bodies extend qpos/qvel, so flattened sim states recorded
from the native model no longer fit ``set_state_from_flattened``.
``set_state`` accepts BOTH layouts: a native-length state is spliced with
the current drop coordinates (drops persist across state restores), an
augmented-length state is applied directly.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
from scene import LAYOUTS

BPP_ROOT = Path(__file__).resolve().parents[2]
TASK = 'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box'
SPLIT = 'libero_spatial_selected_combinations_view'
OBJECT = 'akita_black_bowl_1'
GOAL = 'cookies_1'
OBSTACLE_RADIUS = 0.025
OBSTACLE_HEIGHT = 0.18
WATER_RADIUS = 0.034
WATER_HEIGHT = 0.008
# Local-frame offset of the water surface inside the bowl: near the rim
# (rim-top geoms sit ~85 mm above the bowl body origin) so the vessel reads
# as "full" at BPP input resolution.
WATER_LOCAL_Z = 0.074
# Invisible retainer: a mocap collider glued to the bowl just above the
# fill surface. It collides ONLY with the drops (contype=8/conaffinity=0)
# and caps the fill like a lid — the native demo's own handling (up to
# ~26° lift tilt plus release jolt) ejects any loose fill without it.
# Balls still pour past the rim gap under genuine tip-over (~50°+).
FILL_LOCAL_Z = 0.038
FILL_RADIUS = 0.028
DROP_RADIUS = 0.008
DROP_RGBA = '0.15 0.45 1.00 0.95'
MATERIALS = {'plastic': '0.85 0.23 0.08 1.0',
             'glass': '0.55 0.80 1.00 0.35'}
SOURCE = BPP_ROOT / 'datasets/libero' / SPLIT / (TASK + '_demo.hdf5')
BDDL = (BPP_ROOT / 'src/behavior_prompting/train_network/env/libero/bddl_files'
        / SPLIT / (TASK + '.bddl'))


def state_hash(state):
    return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()


class SemanticScene:
    def __init__(self, resolution=128, record_images=True, water=False,
                 n_drops=0, post_rgba=None):
        self.water = water          # water mocap body exists in the model
        self.water_enabled = water  # whether it is rendered under the bowl
        self.n_drops = n_drops
        self.drops_in_bowl = n_drops > 0  # False -> drops parked on the floor
        self.post_pos = None        # applied to the post mocap at reset
        self.material = None
        self.layout = 'none'
        with h5py.File(SOURCE, 'r') as f:
            demo = f['data/demo_0']
            self.initial_state = demo['states'][0].copy()
            self.reference_seed = int(demo.attrs.get('seed', 2))
        self.wrapper = ControlEnvForRunner(str(BDDL), camera_heights=resolution,
                                           camera_widths=resolution, control_delta=True,
                                           use_camera_obs=record_images,
                                           has_offscreen_renderer=record_images,
                                           ignore_done=True, hard_reset=False)
        self.env = self.wrapper.env
        self.env.seed(self.reference_seed)
        self.env.reset()
        # Reuse the complete installed scene; only append mocap visual bodies.
        tree = ET.fromstring(self.env.sim.model.get_xml())
        world = tree.find('worldbody')
        if water:
            body = ET.SubElement(world, 'body', name='cup_water', mocap='true',
                                 pos='-0.075 0.007 0.92')
            ET.SubElement(body, 'geom', name='cup_water_geom', type='cylinder',
                          size=f'{WATER_RADIUS} {WATER_HEIGHT / 2}',
                          group='1', rgba='0.30 0.62 0.95 0.72',
                          contype='0', conaffinity='0', density='0')
        if post_rgba is not None:
            body = ET.SubElement(world, 'body', name='procedure_obstacle',
                                 mocap='true', pos='0.05 -0.04 0.95')
            ET.SubElement(body, 'geom', name='procedure_obstacle_geom',
                          type='cylinder',
                          size=f'{OBSTACLE_RADIUS} {OBSTACLE_HEIGHT / 2}',
                          group='1', rgba=post_rgba, friction='0.8 0.01 0.001',
                          contype='1', conaffinity='1')
        if n_drops:
            body = ET.SubElement(world, 'body', name='cup_fill',
                                 mocap='true', pos='0 0 -2')
            ET.SubElement(body, 'geom', name='cup_fill_geom', type='cylinder',
                          size=f'{FILL_RADIUS} 0.002', group='1',
                          rgba='0 0 0 0', contype='8', conaffinity='0')
        for i in range(n_drops):
            body = ET.SubElement(world, 'body', name=f'cup_drop_{i}',
                                 pos=f'{1.0 + 0.02 * i} 1.0 {DROP_RADIUS}')
            ET.SubElement(body, 'joint', name=f'cup_drop_{i}_joint',
                          type='free', damping='0.8', frictionloss='0.004')
            ET.SubElement(body, 'geom', name=f'cup_drop_{i}_geom',
                          type='sphere', size=str(DROP_RADIUS),
                          group='1',  # this renderer only draws geom group 1
                          rgba=DROP_RGBA, density='120',
                          friction='0.8 0.05 0.05', margin='0.001',
                          solimp='0.7 0.8 0.001', solref='0.02 1',
                          contype='1', conaffinity='9')
        self.env.reset_from_xml_string(ET.tostring(tree, encoding='unicode'))
        self.env.hard_reset = False
        self.model_xml = self.env.sim.model.get_xml()
        self.m = self.env.sim.model._model
        self.d = self.env.sim.data._data
        # Drop bookkeeping: freejoint adds 7 qpos + 6 qvel at the END of the
        # state vector, so native-length states splice cleanly.
        self.drop_bodies = [self.env.sim.model.body_name2id(f'cup_drop_{i}')
                            for i in range(n_drops)]
        jnts = [self.m.body_jntadr[b] for b in self.drop_bodies]
        self.drop_qposadr = [int(self.m.jnt_qposadr[j]) for j in jnts]
        self.drop_dofadr = [int(self.m.jnt_dofadr[j]) for j in jnts]
        self.drop_geoms = set(self.geoms_for_bodies(self.drop_bodies)) \
            if n_drops else set()
        self.native_flatlen = 1 + (self.m.nq - 7 * n_drops) \
            + (self.m.nv - 6 * n_drops) + self.m.na
        self.obstacle_id = (self.env.sim.model.geom_name2id('procedure_obstacle_geom')
                            if post_rgba is not None else None)
        if water:
            water_body = self.env.sim.model.body_name2id('cup_water')
            self.water_mocap_id = int(self.m.body_mocapid[water_body])
        if n_drops:
            fill_body = self.env.sim.model.body_name2id('cup_fill')
            self.fill_mocap_id = int(self.m.body_mocapid[fill_body])
            self.fill_geom = self.env.sim.model.geom_name2id('cup_fill_geom')
        if self.obstacle_id is not None:
            obody = self.env.sim.model.body_name2id('procedure_obstacle')
            self.mocap_id = int(self.m.body_mocapid[obody])
        self.object_body = self.env.obj_body_id[OBJECT]
        self.goal_body = self.env.obj_body_id[GOAL]
        self.object_geoms = self.geoms_for_bodies([self.object_body])
        self.robot_geoms = [i for i in range(self.m.ngeom)
                            if self.m.geom_bodyid[i] > 0
                            and self.geom_name(i).startswith(('robot0_', 'gripper0_'))
                            and self.collidable(i)]
        self.gripper_geoms = [i for i in self.robot_geoms
                              if self.geom_name(i).startswith('gripper0_')]
        self.record_images = record_images
        self.canonical_state = self.initial_state
        self.reset()

    def collidable(self, i):
        return bool(self.m.geom_contype[i] or self.m.geom_conaffinity[i])

    def geom_name(self, i):
        return self.env.sim.model.geom_id2name(i) or ''

    def geoms_for_bodies(self, ids):
        return [i for i in range(self.m.ngeom)
                if self.m.geom_bodyid[i] in ids and self.collidable(i)]

    def set_state(self, flat):
        """Restore a flattened sim state of EITHER layout: native-length
        states (recorded without drop bodies) are spliced with the current
        drop qpos/qvel so drops persist; augmented states apply directly."""
        if self.n_drops and len(flat) == self.native_flatlen:
            nq_nat = self.m.nq - 7 * self.n_drops
            nv_nat = self.m.nv - 6 * self.n_drops
            dq = np.concatenate([self.d.qpos[a:a + 7]
                                 for a in self.drop_qposadr])
            dv = np.concatenate([self.d.qvel[a:a + 6]
                                 for a in self.drop_dofadr])
            flat = np.concatenate([flat[:1 + nq_nat], dq,
                                   flat[1 + nq_nat:1 + nq_nat + nv_nat], dv,
                                   flat[1 + nq_nat + nv_nat:]])
        self.env.sim.set_state_from_flattened(flat)

    def place_drops(self):
        """Arrange drops inside the level bowl (two heaped layers near the
        rim), or park them on the far floor when the variant is Empty."""
        if not self.n_drops:
            return
        self.env.sim.forward()  # xpos is stale right after set_state
        self.update_water()   # position the fill plug at the bowl first
        self.env.sim.forward()
        if self.drops_in_bowl:
            pos = self.d.xpos[self.object_body]
            # Fill the REAL bowl interior, ~2 layers ending well below the
            # lip: survives the demo's own ~26-degree lift tilt (verified 0
            # spills through a full replay) while a genuine tip-over still
            # pours balls out. Rim-level fills cannot survive the lift.
            pattern = []
            for li, dz in enumerate([0.015, 0.027]):
                ring = [(0., 0., dz)]
                ring += [(0.012 * np.cos(a), 0.012 * np.sin(a), dz)
                         for a in np.linspace(0, 2 * np.pi, 7, endpoint=False)
                         + li * 0.9]
                pattern += ring
            for i, a in enumerate(self.drop_qposadr):
                dx, dy, dz = pattern[i % len(pattern)]
                self.d.qpos[a:a + 3] = [pos[0] + dx, pos[1] + dy, pos[2] + dz]
                self.d.qpos[a + 3:a + 7] = [1., 0., 0., 0.]
        else:
            for i, a in enumerate(self.drop_qposadr):
                self.d.qpos[a:a + 3] = [1.0 + 0.02 * i, 1.0, DROP_RADIUS]
                self.d.qpos[a + 3:a + 7] = [1., 0., 0., 0.]
        for a in self.drop_dofadr:
            self.d.qvel[a:a + 6] = 0.
        self._settle_drops()

    def _settle_drops(self, max_steps=800, check_every=25,
                      vel_tol=1e-3, pos_tol=1e-4):
        """Step the sim until the heap is measurably at rest (drop velocities
        and inter-check drift both below tolerance), so demos and rollouts
        always start from a fully static pile."""
        prev = np.stack([self.d.qpos[a:a + 3]
                         for a in self.drop_qposadr]).copy()
        steps = 0
        while steps < max_steps:
            for _ in range(check_every):
                self.env.sim.step()
            steps += check_every
            v = max(float(np.abs(self.d.qvel[a:a + 6]).max())
                    for a in self.drop_dofadr)
            now = np.stack([self.d.qpos[a:a + 3]
                            for a in self.drop_qposadr])
            drift = float(np.abs(now - prev).max())
            prev = now.copy()
            if v < vel_tol and drift < pos_tol:
                break
        for a in self.drop_dofadr:
            self.d.qvel[a:a + 6] = 0.
        # Re-apply the canonical state: drop coordinates are preserved by the
        # splice, so the robot/bowl come back bit-exact while drops keep their
        # settled positions.
        self.set_state(self.canonical_state)
        self.env.sim.forward()

    def spill_status(self):
        """Per-drop containment: True while the drop sits in the bowl cavity
        (above the interior floor, below just over the rim, inside the
        footprint). Drops riding the gripper or resting on the table count
        as spilled."""
        if not self.n_drops:
            return np.zeros(0, dtype=bool), np.zeros((0, 3))
        pos = self.d.xpos[self.object_body]
        pts = np.stack([self.d.xpos[b] for b in self.drop_bodies])
        if not self.drops_in_bowl:
            # Parked variant: nothing to spill.
            return np.ones(len(pts), dtype=bool), pts
        # Bowl-local containment: rotate into the bowl frame so a tipped
        # bowl still accounts correctly — a drop is "out" once it is past
        # the lip plane or outside the cavity radius in bowl coordinates.
        quat = self.d.xquat[self.object_body].copy()
        lp = Rotation.from_quat(quat, scalar_first=True).inv().apply(
            pts - pos)
        r_xy = np.linalg.norm(lp[:, :2], axis=1)
        inside = (r_xy < 0.045) & (lp[:, 2] > -0.005) & (lp[:, 2] < 0.085)
        return inside, pts

    def spill_count(self):
        inside, _ = self.spill_status()
        return int((~inside).sum()) if len(inside) else 0

    def native_flat(self, flat=None):
        """Flattened state with the drop-body rows removed — the part of the
        state that is comparable between Empty and Filled variants."""
        flat = self.env.sim.get_state().flatten() if flat is None else flat
        if not self.n_drops or len(flat) == self.native_flatlen:
            return flat
        k = self.n_drops
        nq_n = self.m.nq - 7 * k
        nv_n = self.m.nv - 6 * k
        return np.concatenate([flat[:1 + nq_n],
                               flat[1 + nq_n + 7 * k:1 + nq_n + 7 * k + nv_n],
                               flat[1 + self.m.nq + self.m.nv:]])

    def reset(self):
        np.random.seed(self.reference_seed)
        self.env.seed(self.reference_seed)
        self.env.reset()
        self.set_state(self.canonical_state)
        self.env.sim.forward()
        if len(self.canonical_state) == self.native_flatlen:
            self.place_drops()  # augmented states already carry drop coords
        if self.obstacle_id is not None:
            if self.post_pos is not None:
                self.d.mocap_pos[self.mocap_id] = self.post_pos
                self.d.mocap_quat[self.mocap_id] = [1, 0, 0, 0]
                LAYOUTS['post'] = np.asarray(self.post_pos)
            if self.material is not None:
                self.layout = self.material
                LAYOUTS[self.material] = self.d.mocap_pos[self.mocap_id].copy()
            else:
                self.layout = 'post'
        self.update_water()
        self.env.sim.forward()
        self.env.robots[0].controller.update(force=True)
        self.env.robots[0].controller.reset_goal()
        self.env.robots[0].gripper.current_action = np.zeros(2)
        self.env._update_observables(force=True)
        return self.obs()

    def update_water(self):
        """Keep the water disk centered on the bowl and world-level (or park it
        out of sight when the variant is Empty). Purely visual. Also keeps
        the invisible fill plug glued to the bowl's full pose."""
        if self.water:
            if not self.water_enabled:
                self.d.mocap_pos[self.water_mocap_id] = [0., 0., -2.]
            else:
                pos = self.d.xpos[self.object_body].copy()
                quat = self.d.xquat[self.object_body].copy()
                offset = Rotation.from_quat(quat, scalar_first=True).apply(
                    np.array([0., 0., WATER_LOCAL_Z]))
                self.d.mocap_pos[self.water_mocap_id] = pos + offset
                self.d.mocap_quat[self.water_mocap_id] = [1., 0., 0., 0.]
        if self.n_drops:
            if not self.drops_in_bowl:
                self.d.mocap_pos[self.fill_mocap_id] = [0., 0., -2.]
            else:
                pos = self.d.xpos[self.object_body].copy()
                quat = self.d.xquat[self.object_body].copy()
                off = Rotation.from_quat(quat, scalar_first=True).apply(
                    np.array([0., 0., FILL_LOCAL_Z]))
                self.d.mocap_pos[self.fill_mocap_id] = pos + off
                self.d.mocap_quat[self.fill_mocap_id] = quat

    def obs(self):
        return self.env._get_observations(force_update=True)

    def object_pos(self):
        return self.d.xpos[self.object_body].copy()

    def object_quat(self):
        return self.d.xquat[self.object_body].copy()

    def goal_pos(self):
        return self.d.xpos[self.goal_body].copy()

    def initial_identity(self):
        state = self.env.sim.get_state().flatten()
        ident = {'sim_state_sha256': state_hash(state),
                 'gripper_command_state': self.env.robots[0].gripper.current_action.tolist(),
                 'controller_goal_pos': self.env.robots[0].controller.goal_pos.tolist(),
                 'controller_goal_ori': self.env.robots[0].controller.goal_ori.tolist()}
        if self.obstacle_id is not None:
            ident['obstacle_pos'] = self.d.mocap_pos[self.mocap_id].tolist()
        if self.water:
            ident['water'] = 'visual-mocap (no collision, level surface)'
        if self.n_drops:
            ident['drops'] = (f'{self.n_drops} free spheres r={DROP_RADIUS}m '
                              + ('in bowl' if self.drops_in_bowl else 'parked')
                              + ' - real dynamics, spillable')
        return ident

    def surface_clearance(self, geoms):
        if self.obstacle_id is None:
            return float('inf')
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
            if pair & self.drop_geoms:
                continue
            if (a in gripper and b in obj) or (b in gripper and a in obj):
                continue
            if self.obstacle_id is not None and self.obstacle_id in pair \
                    and pair & (robot | obj):
                collisions.append((self.geom_name(a), self.geom_name(b), float(c.dist)))
            elif a in robot and b in robot and not (a in gripper and b in gripper):
                collisions.append((self.geom_name(a), self.geom_name(b), float(c.dist)))
            elif (a in robot) != (b in robot):
                other = b if a in robot else a
                if self.geom_name(other) not in ('floor',):
                    collisions.append((self.geom_name(a), self.geom_name(b), float(c.dist)))
            elif transport and ((a in obj) != (b in obj)) and not pair & gripper:
                collisions.append((self.geom_name(a), self.geom_name(b), float(c.dist)))
        return collisions

    def close(self):
        self.wrapper.close()
