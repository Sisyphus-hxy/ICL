"""Render validated simulation states without rerunning expert control/physics."""
import hashlib
import json

import h5py
import numpy as np

from behavior_prompting.train_network.utils.video_recorder import VideoRecorder
from scene import LAYOUTS, Scene


def render_experts(output, stride=4):
    # 5 Hz video is sufficient for inspection. The actual BPP prompt samples
    # every 20 control steps (1 Hz), so every prompt frame is rendered exactly.
    assert 20 % stride == 0
    scene=Scene()
    identities={}
    try:
        (output/'scene.xml').write_text(scene.model_xml)
        for layout in ['original','relocated']:
            for procedure in ['narrow','wide']:
                name=f'expert_{layout}_{procedure}'
                path=output/(name+'.hdf5')
                if not path.is_file():
                    continue
                if path.is_symlink():
                    raise RuntimeError(f'Refusing to rewrite a reused expert file: {path}')
                print('Rendering visible obstacle:',name,flush=True)
                scene.reset(layout)
                with h5py.File(path,'r+') as f:
                    d=f['data/demo_0'];states=d['states'][:]
                    if d.attrs.get('observation_alignment') != 'post_action' or 'terminal_state' not in d:
                        raise ValueError('Regenerate legacy experts: their pre-action observations do not match the official dataset.')
                    # Retain the true reset image for pairing checks. Prompt
                    # frame t must instead show the state after action t.
                    initial_obs=scene.obs()
                    identity=json.loads(d.attrs['initial_identity'])
                    identity['initial_rgb_sha256']=hashlib.sha256(
                        initial_obs['agentview_image'].tobytes()+
                        initial_obs['robot0_eye_in_hand_image'].tobytes()).hexdigest()
                    identities[name]=identity
                    post_states=np.concatenate([states[1:],d['terminal_state'][:][None]],axis=0)
                    video=VideoRecorder.create_h264(fps=20//stride,thread_count=1)
                    video.start(str(output/(name+'.mp4')))
                    images={key:[] for key in ['agentview_rgb','eye_in_hand_rgb']}
                    try:
                        for step in range(0,len(post_states),stride):
                            scene.env.sim.set_state_from_flattened(post_states[step])
                            scene.d.mocap_pos[scene.mocap_id]=LAYOUTS[layout]
                            scene.env.sim.forward()
                            obs=scene.obs()
                            left=obs['agentview_image'];right=obs['robot0_eye_in_hand_image']
                            images['agentview_rgb'].append(left.copy())
                            images['eye_in_hand_rgb'].append(right.copy())
                            video.write_frame(np.concatenate([left[::-1],right[::-1]],axis=1).copy())
                    finally:
                        video.stop()
                    for key,values in images.items():
                        if key in d['obs']:
                            del d['obs'][key]
                        dense=np.repeat(np.asarray(values),stride,axis=0)[:len(states)]
                        d['obs'].create_dataset(key,data=dense,compression='gzip',compression_opts=1)
                    d.attrs['model_file']=scene.model_xml
                    d.attrs['initial_identity']=json.dumps(identities[name])
                    d.attrs['rgb_capture_stride_control_steps']=stride
                    d.attrs['rgb_storage']='post-action 5Hz zero-order hold; every 1Hz BPP prompt frame is exact'
        assert identities['expert_original_narrow']['initial_rgb_sha256']==identities['expert_original_wide']['initial_rgb_sha256']
        assert identities['expert_original_narrow']['initial_rgb_sha256']!=identities['expert_relocated_narrow']['initial_rgb_sha256'], 'Obstacle relocation is invisible in RGB'
        (output/'render_validation.json').write_text(json.dumps({'valid':True,'obstacle_visible':True,
            'stride':stride,'video_fps':20//stride,'identities':identities},indent=2))
        return identities
    finally:
        scene.close()
