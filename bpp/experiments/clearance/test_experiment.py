"""Offline contract checks; run with the installed BPP Python environment."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('PYNPUT_BACKEND','dummy')
os.environ.setdefault('EGL_PLATFORM','surfaceless')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
os.environ.setdefault('LP_NUM_THREADS','2')

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from expert import transport_paths, path_length
from inference import load_prompt, query_observations


class ExperimentContracts(unittest.TestCase):
    def test_recorded_observations_follow_official_post_action_convention(self):
        from recording import Recorder
        from scene import Scene
        s=Scene(resolution=32,record_images=False)
        try:
            with tempfile.TemporaryDirectory() as temp:
                rec=Recorder(Path(temp),'alignment',save_images=False)
                rec.start(s)
                initial=rec.latest_obs['robot0_eef_pos'].copy()
                expected=[]
                for _ in range(3):
                    rec.step(np.array([.3,0.,0.,0.,0.,0.,-1.]))
                    expected.append(rec.latest_obs['robot0_eef_pos'].copy())
                terminal=s.env.sim.get_state().flatten().copy()
                rec.finish(s)
                with h5py.File(Path(temp)/'alignment.hdf5','r') as f:
                    demo=f['data/demo_0']
                    self.assertEqual(demo.attrs['observation_alignment'],'post_action')
                    np.testing.assert_array_equal(demo['obs/ee_pos'][:],expected)
                    np.testing.assert_array_equal(demo['initial_ee_pos'][:],initial)
                    np.testing.assert_array_equal(demo['terminal_state'][:],terminal)
                    self.assertFalse(np.array_equal(demo['obs/ee_pos'][0],initial))
        finally:
            s.close()

    def test_disjoint_transport_intervals_do_not_create_fictitious_path(self):
        from recording import path_metrics
        pos=np.array([[0,0,0],[1,0,0],[10,0,0],[11,0,0]])
        metrics=path_metrics(pos,np.array([False,True,False,True]),np.array([-1,0,0]))
        self.assertEqual(metrics['path_length_m'],12.)
        self.assertEqual(metrics['transport_path_length_m'],2.)

    def test_obstacle_relocation_is_visible_and_same_layout_is_identical(self):
        from scene import Scene
        s=Scene(resolution=64)
        try:
            s.reset('original');a=s.obs()['agentview_image'].copy()
            identity=s.initial_identity()
            state_a=s.initial_identity()['sim_state_sha256']
            s.env.robots[0].gripper.current_action=np.array([-1.,1.])
            s.reset('original');b=s.obs()['agentview_image'].copy()
            self.assertEqual(identity,s.initial_identity())
            np.testing.assert_array_equal(a,b)
            s.reset('relocated');c=s.obs()['agentview_image'].copy()
            self.assertEqual(state_a,s.initial_identity()['sim_state_sha256'])
            self.assertFalse(np.array_equal(a,c))
            self.assertEqual(int(s.m.geom_group[s.obstacle_id]),1)
        finally:
            s.close()

    def test_transport_matches_endpoints_length_duration_and_speed(self):
        start=np.array([-.18,.05,1.075]); end=np.array([.14,.065,1.075])
        for layout in ['original','relocated']:
            paths=transport_paths(start,end,layout)
            self.assertEqual(paths['narrow'].shape,paths['wide'].shape)
            for path in paths.values():
                np.testing.assert_allclose(path[0],start,atol=1e-10)
                np.testing.assert_allclose(path[-1],end,atol=1e-10)
                speed=np.linalg.norm(np.diff(path,axis=0),axis=1)
                self.assertLess(speed.std()/speed.mean(),.04)
            self.assertLess(abs(path_length(paths['narrow'])/path_length(paths['wide'])-1),.005)

    def test_procedure_name_is_never_part_of_prompt_tensor(self):
        from hydra import compose, initialize_config_dir
        root=Path(__file__).resolve().parents[2]
        with initialize_config_dir(config_dir=str(root/'src/behavior_prompting/train_network/config'),version_base=None):
            cfg=compose(config_name='libero_policy_dunetp',overrides=['task=liberogen_spatial_combination'])
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'narrow.hdf5'
            with h5py.File(source,'w') as f:
                d=f.create_group('data/demo_0');d.attrs['procedure']='Narrow'
                d.create_dataset('actions',data=np.zeros((40,7)))
                o=d.create_group('obs')
                for key,dim in [('ee_pos',3),('ee_ori',3),('gripper_states',2)]:
                    o.create_dataset(key,data=np.zeros((40,dim)))
                for key in ['agentview_rgb','eye_in_hand_rgb']:
                    o.create_dataset(key,data=np.zeros((40,32,32,3),dtype=np.uint8))
            first=load_prompt(source,cfg.task.shape_meta,'cpu')
            renamed=Path(temp)/'wide.hdf5';source.rename(renamed)
            with h5py.File(renamed,'r+') as f: f['data/demo_0'].attrs['procedure']='Wide'
            second=load_prompt(renamed,cfg.task.shape_meta,'cpu')
            self.assertEqual(set(first),{'obs','action','metadata'})
            self.assertEqual(set(first['metadata']),{'mask'})
            self.assertTrue(torch.equal(first['action'],second['action']))
            for key in first['obs']:
                self.assertTrue(torch.equal(first['obs'][key],second['obs'][key]))
            self.assertEqual(tuple(first['action'].shape),(1,2,20,10))
            with h5py.File(renamed,'r+') as f:
                f['data/demo_0'].attrs['initial_identity']='{}'
            with self.assertRaisesRegex(ValueError,'pre-action'):
                load_prompt(renamed,cfg.task.shape_meta,'cpu')


if __name__=='__main__':
    unittest.main()
