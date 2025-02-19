#%%
import gymnasium as gym
import mani_skill.envs
import time
from mani_skill.utils.wrappers import CPUGymWrapper
import matplotlib.pyplot as plt
import torch
import tqdm
import numpy as np
from IPython.display import Video

from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset
from mani_skill.utils.io_utils import load_json
from mani_skill.trajectory.utils import index_dict, dict_to_list_of_dicts
from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.wrappers.record_zarr import RecordEpisodeZarr

import multiprocessing
from pathlib import Path

import cv2

import time

import einops

from mani_skill.utils.teleoperation import SpacemouseInput
import sys
#%%
# add FISH directory to the path
path_to_fish = Path('~/fish_leon').expanduser()
sys.path.append(str(path_to_fish))
from FISH.agent.diffusion_policy import DiffusionPolicyAgent, DiffusionPolicyAgentConfig
#%%
from hydra import compose, initialize
from omegaconf import OmegaConf

with initialize(version_base=None, config_path="cfgs", job_name="test_app"):
    omegaconf_cfg = compose(config_name="config_eval", overrides=["agent=diffusion"])

print(OmegaConf.to_yaml(omegaconf_cfg))
#%%
def make_agent(cfg):
	dataset_statistics = None # this will be loaded from the checkpoint
	return hydra.utils.instantiate(cfg, dataset_statistics)

#%%
class DiffusionPolicyManiskillWrapper:
    def __init__(self, 
    # policy: DiffusionPolicyAgent, 
    cfg,
    segmentation_id_map: dict, 
    sim_control_freq: int, 
    num_envs: int
    ):
        # >>>>>>>>>> make diffusion policy using omegaconf
        self.cfg = cfg
        snapshot_path = Path(self.cfg.checkpoint_weight_dir) / f'snapshot_{self.cfg.checkpoint_epoch}.pt' 
        self.load_checkpoint_conf(snapshot_path=snapshot_path)

        self.policy = make_agent(cfg)

        if not self.loading_uncompiled_checkpoint_with_compile and self.cfg.agent.config.compile:
            self.policy.compile_modules()
			
        self.load_checkpoint(snapshot_path=snapshot_path)

        if self.loading_uncompiled_checkpoint_with_compile: # need to call compile after loading the checkpoint
            self.policy.compile_modules()

        self.policy.train(False)
        # <<<<<<<<<< make diffusion policy using omegaconf


        # self.policy = policy
        self.device = self.policy.device
        self.policy_config = self.policy.config
        self.segmentation_id_map = segmentation_id_map

        self.action_dim = self.policy_config.policy_cfg.output_shapes['action'][0]
        self.use_action_history = self.policy_config.use_action_history
        if self.use_action_history:
            self.action_history = torch.zeros(num_envs, self.policy_config.policy_cfg.action_history_encoder_config.history_length, self.action_dim, dtype=torch.float32).to(self.device)
            # keep gripper closed (-1)
            self.action_history[:, :, -1] = -1
     
        self.elapsed_steps = 0

        self.policy_freq = self.policy_config.policy_frequency
        self.sim_control_freq = sim_control_freq
        self.replan_in_n_steps = self.sim_control_freq // self.policy_freq

    def maniskill_obs_to_lerobot_obs(obs):
        # TODO: actually implement sequence length handling. For now, we'll just hardcode it in here
        # images should be BxSxCxHxW
        lerobot_batch = {}
        # lerobot_batch['observation.image'] = torch.as_tensor(observations['pixels'], device=self.config.device).float().unsqueeze(0)
        if self.config.observation_cfg.crop_input_config.enable:
            # lerobot_batch['observation.EE_pixel_coord'] = torch.as_tensor(observations['EE_pose_pxl'], device=device).float().unsqueeze(0) # becomes BxSx2
            lerobot_batch['observation.EE_pixel_coord'] = obs['extra']['end_effector_pixel_coord'].unsqueeze(1).to(self.device)
        if self.config.observation_cfg.use_color or self.config.observation_cfg.crop_input_config.color_crop_type != None:
            # BXHXWX3
            lerobot_batch['observation.rgb'] = einops.rearrange(obs['sensor_data']['base_camera']['rgb'].float().unsqueeze(1), 'b s h w c -> b s c h w').to(self.device)
        if self.config.observation_cfg.use_depth or self.config.observation_cfg.crop_input_config.depth_crop_type != None:
            # lerobot_batch['observation.depth'] = pixels[:, :, self.visual_feature_preprocessor.depth_channel_idxs] # BxSx1xHxW
            lerobot_batch['observation.depth'] = einops.rearrange(obs['sensor_data']['base_camera']['depth'].float().unsqueeze(1), 'b s h w c -> b s c h w').to(self.device)
        if self.config.observation_cfg.mask_input_dict.enable or self.config.observation_cfg.crop_input_config.segmask_crop_type != None:
            # lerobot_batch['observation.EE_obj_mask'] = pixels[:, :, self.visual_feature_preprocessor.segmask_channel_idxs]
            grasped_book_mask = self.extract_individual_segmentation_mask(obs['sensor_data']['base_camera']['segmentation'], self.segmentation_id_map, 'grasped_book_0')
            lerobot_batch['observation.EE_obj_mask'] = einops.rearrange(grasped_book_mask.float().unsqueeze(1), 'b s h w c -> b s c h w').to(self.device)
        if self.config.observation_cfg.use_contact_feature:
            # lerobot_batch['observation.contact_map'] = pixels[:, :, self.visual_feature_preprocessor.contact_channel_idxs] # BxSx1xHxW
            # lerobot_batch['observation.env_dtc_map'] = pixels[:, :, self.visual_feature_preprocessor.env_dtc_channel_idxs] # BxSx1xHxW
            # lerobot_batch['observation.EE_dtc_map'] = pixels[:, :, self.visual_feature_preprocessor.grasped_dtc_channel_idxs] # BxSx1xHxW
            # lerobot_batch['observation.EE_normals_map'] = pixels[:, :, self.visual_feature_preprocessor.grasped_normals_channel_idxs] # BxSx3xHxW
            # lerobot_batch['observation.env_normals_map'] = pixels[:, :, self.visual_feature_preprocessor.env_normals_channel_idxs] # BxSx3xHxW
            lerobot_batch['observation.contact_map'] = einops.rearrange(obs['extra']['extrinsic_contact_map'].unsqueeze(1), 'b s h w c -> b s c h w').to(self.device)
        # if action_history is not None:  
        if self.use_action_history:
            # assert action_history is not None and action_history_start_timestamp is not None, "policy uses action history but it was not provided!"
            # lerobot_batch['observation.action_history'] = torch.as_tensor(action_history, device=device).float().unsqueeze(0) # add batch dimension
            lerobot_batch['observation.action_history'] = self.action_history.unsqueeze(1) # BxTxA to BxSxTxA 
        # lerobot_batch['observation.state'] = torch.as_tensor(observations['features'], device=self.config.device).float().unsqueeze(0) # add batch dimension
        pose = obs['extra']['end_effector_pose'] # bx7
        gripper_width = torch.sum(obs['agent']['qpos'][:,-2:], dim=-1, keepdim=True) #bx1
        # concatenate pose and gripper width to get bx8
        pose_gripper = torch.cat([pose, gripper_width], dim=-1)
        lerobot_batch['observation.state'] = pose_gripper.unsqueeze(1).to(self.device)
        return lerobot_batch
    
    @staticmethod
    def extract_individual_segmentation_mask(segmentation_map, segmentation_id_map, id):
        mask = (segmentation_map == id).float()
        assert len(mask.shape) == 4, f"mask should be BxHxWx1, but got {mask.shape}"
        assert mask.shape[-1] == 1, f"mask should be BxHxWx1, but got {mask.shape}"
        return 

    def reset(self):
        self.current_action_plan = None
        if self.use_action_history:
            self.action_history[...] = 0
            self.action_history[..., -1] = -1
        self.elapsed_steps = 0

    def act(self, maniskill_obs):
        if self.elapsed_steps % self.replan_in_n_steps == 0:
            lerobot_obs = self.maniskill_obs_to_lerobot_obs(maniskill_obs)
            action_plan = self.policy.act(lerobot_obs) # BxTxA
            self.current_action_plan = action_plan
        assert self.current_action_plan is not None, "current_action_plan is None"
        # get the action to be executed from the current_action_plan
        action_to_execute = self.current_action_plan[:, self.elapsed_steps % self.replan_in_n_steps]
        if self.use_action_history:
            # THIS IS TECHNICALLY WRONG AS EXECUTED ACTIONS SHOULD BE FILLED INTO A QUEUE FROM THE RIGHT SIDE RATHER THAN FILL FROM LEFT
            # this only works for now because history length is exactly the same as the replan_in_n_steps
            self.action_history[:, self.elapsed_steps % self.replan_in_n_steps] = action_to_execute
        self.elapsed_steps += 1
        return action_to_execute
    
    def load_checkpoint_conf(self, snapshot_path):
        config_path = snapshot_path.parent / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f'No snapshot conf found at {config_path}')
        else:
            # load the omegaconf config
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            hydra.initialize(
                str(_relative_path_between(Path(config_path).absolute().parent, Path(__file__).absolute().parent)),
            )
            cfg = hydra.compose(Path(config_path).stem)
            from deepdiff import DeepDiff
            from omegaconf import open_dict
            diff = DeepDiff(OmegaConf.to_container(cfg), OmegaConf.to_container(self.cfg)) # old, new
            # import re
            overwriteable_keys = [f"root{overwritable_key}" for overwritable_key in ["['path_to_depth_extrinsics']", "['eval']", "['root_dir']", "['wandb_notes']", "['agent']['config']['train_cfg']['use_amp']", "['agent']['config']['compile']", "['agent']['config']['policy_cfg']['num_inference_steps']"]]
            if "values_changed" in diff:
                # top_k_checkpoints, wandb_notes, agent.config.train_cfg.use_amp, save_snapshot_every_epochs_diffusion, check_topk_every_epochs_diffusion, validate_diffusion_on_action_loss_every_epochs, train_eval_diffusion_on_action_loss_every_epochs, validate_every_epochs_diffusion
                # for keys above, overwrite the old config with the new config
                for k, v in diff['values_changed'].items():
                    # replace any keys that are under "root['suite']"
                    if k in overwriteable_keys or k.startswith("root['suite']"):
                        print(f"Found changed key {k} with value {v}. Overwriting old checkpoint config")
                        if k == "root['agent']['config']['compile']":
                            if diff['values_changed'][k]['new_value']:
                                self.loading_uncompiled_checkpoint_with_compile = True
                            elif not diff['values_changed'][k]['new_value']:
                                # raise ValueError("Cannot load a compiled checkpoint without compile")
                                self.loading_compiled_checkpoint_with_no_compile = True
                        exec(f"{k.replace('root[', 'cfg[')} = {k.replace('root[', 'self.cfg[')}")
            # for any new values, update the old checkpoint config
            if "dictionary_item_added" in diff:
                for new_key in diff['dictionary_item_added']: # this is a list
                    # if new_key == "root['suite']['task_make_fn']['observation_cfg']":
                    if new_key == "root['suite']['task_make_fn']['agent_policy_cfg']":
                        # pass the agents observation_cfg to the suite task_make_fn
                        with open_dict(cfg): # to allow addition of non-existing keys
                            cfg.suite.task_make_fn.observation_cfg = cfg.agent.config.observation_cfg
                        continue
                    elif "['agent']['config']['policy_cfg']['input_shapes']" in new_key:
                        # skip adding the new key if it is the input_shapes of the policy_cfg
                        continue
                    else:
                        print(f"Found new key {new_key} with value {eval(new_key.replace('root[', 'self.cfg['))}. Adding to checkpoint config")
                    # eval(new_key.replace('root', 'cfg')) = eval(new_key.replace('root', 'self.cfg'))
                    if new_key == "root['agent']['config']['compile']":
                        if self.cfg.agent.config.compile:
                            self.loading_uncompiled_checkpoint_with_compile = True
                    
                    with open_dict(cfg):
                        exec(f"{new_key.replace('root[', 'cfg[')}={new_key.replace('root[', 'self.cfg[')}")
            self.cfg = cfg

    def load_checkpoint(self, snapshot_path, bc=False):
        print(f'resuming {repr(self.policy)}: {snapshot_path}')
        with snapshot_path.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
            elif k == '_global_epoch':
                self._global_epoch = v
                print(f'loaded epoch: {v}')
                if self.cfg.use_wandb:
                    # add to config of wandb
                    wandb.config.update({'epoch': v})

        self.policy.load_snapshot_eval(agent_payload, bc)
#%%
# spacemouse_input = SpacemouseInput()
desired_viewing_size = (256, 256)

#%%
## testing book insertion task
env = gym.make(
    # "LiftPegUpright-v1", 
    "BookInsertion-v0", 
    cam_resize_factor=0.5,
    reward_mode="none", 
    sim_backend='physx_cpu', 
    # render_mode="rgb_array", 
    # render_mode="sensors", 
    obs_mode="rgb+depth+segmentation",
    # obs_mode="none",
    control_mode="pd_ee_target_delta_pose",
    # control_mode="pd_ee_delta_pose",
    sim_config=dict(
        sim_freq=100, # default 100
        control_freq=20, # default 20
        scene_config=dict(
            solver_position_iterations=15, # 15 is the default
            contact_offset=0.02, # 0.02 is the default
            # contact_offset=0.02, # 0.02 is the default
            cpu_workers=0, # 0 is the default
        )
    ),
    viewer_camera_configs=dict(
        shader_pack="minimal"
    ),
    human_render_camera_configs=dict(
        shader_pack="minimal"
    )
)
seed = 0
num_trajs = 0
#%%
sim_dt = 1.0 / env.sim_config.sim_freq
sim_dt_bw_step = sim_dt * (env.sim_config.sim_freq / env.sim_config.control_freq)

human_render_cam_params = env.scene.human_render_cameras['render_camera'].get_params()
human_render_cam_intrisic = human_render_cam_params['intrinsic_cv'][0]
human_render_cam_cam2world_gl = human_render_cam_params['cam2world_gl'][0][:3, :4]
human_render_cam_extrinsic_cv = human_render_cam_params['extrinsic_cv'][0]
#%%
base_camera_cam2world_gl = env.scene.sensors['base_camera'].get_params()['cam2world_gl'][0]
base_camera_extrinsic_cv = env.scene.sensors['base_camera'].get_params()['extrinsic_cv'][0]
#%%
# make the policy
diffusion_polcy = DiffusionPolicyManiskillWrapper(omegaconf_cfg, env.segmentation_id_map, env.sim_config.control_freq, env.num_envs)

#%% 
obs, info = env.reset(seed=seed)
action = diffusion_polcy.act(obs)
#%%
# frame = obs['sensor_data']['base_camera']['segmentation'][0].cpu().numpy()
# frame = env.render_rgb_array()[0].cpu().numpy()
# plt.imshow(frame)
# plt.imshow(obs['sensor_data']['base_camera']['Color'][0][:,:,3].cpu().numpy())

#%%
# cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)

# frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
# frame = (frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()*255*0.5).astype(np.uint8)
# frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
# frame = cv2.cvtColor(env.render()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)

# frame = cv2.resize(frame, desired_viewing_size, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("frame", frame)
# plt.imshow(frame)

viewer = env.render_human()
# viewer.paused = True
#%%
# frames = [env.render_rgb_array()[0].cpu().numpy()]
# for i in tqdm.tqdm(range(500)):
while True:
    start_time = time.perf_counter()
    while True:
        # action = env.action_space.sample()
        # action = spacemouse_input.get_action()
        obs, reward, terminated, truncated, info = env.step(action)

        env.render_human()

        action = diffusion_polcy.act(obs)

        # current_frame = cv2.cvtColor(env.render_rgb_array()[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
        # current_frame = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        # current_frame = obs['sensor_data']['base_camera']['Color'][0][:,:,:3].cpu().numpy()

        # current_frame = (current_frame*0.5 + obs['extra']['extrinsic_contact_map'][0].cpu().numpy()*255*0.5).astype(np.uint8)
        # current_frame = cv2.resize(current_frame, desired_viewing_size, interpolation=cv2.INTER_NEAREST)

        # cv2.imshow("frame", current_frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q') or key == ord('c') or key == ord('r'):
        #     break
        
        if viewer.window.key_press('q'):
            # q: quit the script and stop collecting data. Save trajectories and optionally videos.
            # c: stop this episode and record the trajectory and move on to a new episode
            # r: restart
            key = ord('q')
            break
        elif viewer.window.key_press('c'): 
            key = ord('c')
            break
        elif viewer.window.key_press('r'):
            key = ord('r')
            break

        # frames.append(current_frame)
        elapsed_timesteps = info["elapsed_steps"].item()
        elapsed_simtime = elapsed_timesteps * sim_dt_bw_step
        elapsed_realtime = time.perf_counter() - start_time
        # time_to_sleep = sim_dt_bw_step - elapsed_time
        time_to_sleep = elapsed_simtime - elapsed_realtime
        # if time_to_sleep > 0:
        #     time.sleep(time_to_sleep)
        if elapsed_timesteps % 100 == 0:
            print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {elapsed_timesteps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
            print(f"success: {info['success']} | success duration: {info['elapsed_success_duration']} | t. success: {info['transient_success']} | z_distance: {info['z_distance_bw_top_of_grasped_book_and_top_of_slot']}")
    
    if key == ord('q'):
        num_trajs += 1
        break
    elif key == ord('c'):
        seed += 1
        num_trajs += 1
        env.reset(seed=seed)
        diffusion_polcy.reset()
        action = diffusion_polcy.act(obs)
        viewer = env.render_human()
        # spacemouse_input.reset()
        continue
    elif key == ord('r'):
        env.reset(seed=seed, options=dict(save_trajectory=False))
        action = diffusion_polcy.act(obs)
        viewer = env.render_human()
        # spacemouse_input.reset()
        continue
    else:
        break

cv2.destroyAllWindows()
#%%
# if record_demonstrations:
#     h5_file_path = env._h5_file.copy
#     json_file_path = env._json_path

env.close()
del env

# spacemouse_input.close()

# TODO: try adding contact map to extra states and the end effector pose (to get back observation.state)