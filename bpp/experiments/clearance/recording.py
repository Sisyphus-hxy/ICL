"""Physics-step collision/clearance measurement and portable experiment artifacts."""
import csv
import json

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from behavior_prompting.train_network.utils.video_recorder import VideoRecorder
from scene import LAYOUTS


def path_metrics(positions, transport, initial):
    travel=np.linalg.norm(np.diff(np.vstack([initial,positions]),axis=0),axis=1)
    selected=travel[transport]
    return {'path_length_m':float(travel.sum()),
            'transport_path_length_m':float(selected.sum()) if len(selected) else None,
            'transport_speed_mean_m_s':float(selected.mean()*20) if len(selected) else None}


class Recorder:
    def __getattr__(self, name):
        return getattr(self.scene.env, name)

    def __init__(self, directory, name, save_images=True):
        self.directory = directory
        self.name = name
        self.save_images = save_images
        self.phase = 'unknown'
        self.rows, self.actions, self.states, self.observations = [], [], [], []
        self.gripper_commands=[]
        self.collision_events = []
        self.substep_min = {'object': 2., 'gripper': 2.}
        self.phase_collision = False
        self.video = None

    def start(self, scene):
        self.scene = scene
        self.identity = scene.initial_identity()
        self.latest_obs = scene.obs()
        if self.save_images:
            import hashlib
            self.identity['initial_rgb_sha256'] = hashlib.sha256(
                self.latest_obs['agentview_image'].tobytes() +
                self.latest_obs['robot0_eye_in_hand_image'].tobytes()).hexdigest()
        self._physics_step = scene.env.sim.step
        scene.env.sim.step = self._monitored_step
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            self.video = VideoRecorder.create_h264(fps=20, thread_count=1)
            self.video.start(str(self.directory / (self.name + '.mp4')))

    def _monitored_step(self):
        self._physics_step()
        s = self.scene
        for key, geoms in [('object', s.object_geoms), ('gripper', s.gripper_geoms)]:
            self.substep_min[key] = min(self.substep_min[key], s.surface_clearance(geoms))
        collisions = s.contacts(transport=self.phase == 'transport')
        if collisions:
            self.phase_collision = True
            self.collision_events.append({'control_step': len(self.actions), 'phase': self.phase,
                                          'sim_time': float(s.d.time), 'contacts': collisions})

    def _get_observations(self):
        return self.latest_obs

    def step(self, action):
        s = self.scene
        obs = self.latest_obs
        self.observations.append({k: obs[k].copy() for k in ['robot0_eef_pos', 'robot0_eef_quat',
                                  'robot0_gripper_qpos', 'agentview_image', 'robot0_eye_in_hand_image']
                                  if k in obs})
        self.states.append(s.env.sim.get_state().flatten().copy())
        self.gripper_commands.append(s.env.robots[0].gripper.current_action.copy())
        self.actions.append(np.asarray(action).copy())
        if self.video:
            frame = np.concatenate([obs['agentview_image'][::-1], obs['robot0_eye_in_hand_image'][::-1]], axis=1)
            self.video.write_frame(frame.copy())
        self.substep_min = {'object': 2., 'gripper': 2.}
        self.phase_collision = False
        result = s.env.step(np.asarray(action))
        self.latest_obs = result[0]
        p = self.latest_obs['robot0_eef_pos']
        obj = s.object_pos()
        ob = LAYOUTS[s.layout]
        self.rows.append(dict(step=len(self.actions)-1, time_s=len(self.actions)/20, phase=self.phase,
                              eef_x=p[0], eef_y=p[1], eef_z=p[2],
                              object_x=obj[0], object_y=obj[1], object_z=obj[2],
                              object_rel_x=obj[0]-ob[0], object_rel_y=obj[1]-ob[1], object_rel_z=obj[2]-ob[2],
                              object_clearance_m=self.substep_min['object'],
                              gripper_clearance_m=self.substep_min['gripper'],
                              collision=int(self.phase_collision), goal_success=int(s.env._check_success())))
        return result

    def finish(self, scene):
        scene.env.sim.step = self._physics_step
        if self.video:
            self.video.stop()
        with (self.directory / (self.name + '_trajectory.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)
        with (self.directory / (self.name + '_contacts.json')).open('w') as f:
            json.dump(self.collision_events, f, indent=2)
        with h5py.File(self.directory / (self.name + '.hdf5'), 'w') as f:
            data = f.create_group('data')
            demo = data.create_group('demo_0')
            demo.attrs['model_file'] = scene.model_xml
            demo.attrs['seed'] = scene.reference_seed
            demo.attrs['initial_identity'] = json.dumps(self.identity)
            demo.attrs['num_samples'] = len(self.actions)
            # Match upstream scripts/libero/create_dataset.py: obs[t] is the
            # observation returned by step(actions[t]); states[t] is pre-step.
            demo.attrs['observation_alignment'] = 'post_action'
            demo.attrs['state_alignment'] = 'pre_action'
            demo.create_dataset('states', data=np.asarray(self.states))
            demo.create_dataset('terminal_state',data=scene.env.sim.get_state().flatten())
            demo.create_dataset('initial_ee_pos',data=self.observations[0]['robot0_eef_pos'])
            demo.create_dataset('actions', data=np.asarray(self.actions))
            demo.create_dataset('gripper_command_state',data=np.asarray(self.gripper_commands))
            obs_group = demo.create_group('obs')
            mapping = {'robot0_eef_pos': 'ee_pos', 'robot0_gripper_qpos': 'gripper_states',
                       'agentview_image': 'agentview_rgb', 'robot0_eye_in_hand_image': 'eye_in_hand_rgb'}
            post_observations = self.observations[1:] + [self.latest_obs]
            for sim_key, key in mapping.items():
                if sim_key in post_observations[0]:
                    obs_group.create_dataset(key, data=np.asarray([o[sim_key] for o in post_observations]),
                                             compression='gzip', compression_opts=1)
            ori = Rotation.from_quat(np.asarray([o['robot0_eef_quat'] for o in post_observations])).as_rotvec()
            obs_group.create_dataset('ee_ori', data=ori)
            demo.create_dataset('phases', data=np.asarray([r['phase'].encode() for r in self.rows]))
        return self.summary()

    def summary(self):
        pos = np.asarray([[r[k] for k in ['eef_x', 'eef_y', 'eef_z']] for r in self.rows])
        transport = np.asarray([r['phase'] == 'transport' for r in self.rows])
        indices = np.where(transport)[0]
        paths=path_metrics(pos,transport,self.observations[0]['robot0_eef_pos'])
        return {'name': self.name, 'goal_success': bool(self.rows[-1]['goal_success']),
                'collision': any(r['collision'] for r in self.rows),
                'collision_control_steps': sum(r['collision'] for r in self.rows),
                'object_min_clearance_m': min(r['object_clearance_m'] for r in self.rows if r['phase'] == 'transport')
                    if len(indices) else None,
                'gripper_min_clearance_m': min(r['gripper_clearance_m'] for r in self.rows if r['phase'] == 'transport')
                    if len(indices) else None,
                **paths,
                'control_steps': len(self.rows), 'duration_s': len(self.rows)/20,
                'initial_identity': self.identity,
                'evaluation_regime': 'zero-shot diagnostic'}
