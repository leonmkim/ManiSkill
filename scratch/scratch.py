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
import h5py

import cv2
#%%
!python -m mani_skill.utils.download_demo "PegInsertionSide-v1" -o demos
#%%
dataset = ManiSkillTrajectoryDataset(dataset_file="demos/PegInsertionSide-v1/motionplanning/trajectory.h5")

#%%
episode_idx = 10
traj_path = f"demos/PegInsertionSide-v1/motionplanning/trajectory.h5"
# You can also replace the above path with the trajectory you just recorded (./tmp/trajectory.h5)
h5_file = h5py.File(traj_path, "r")

# Load associated json
json_path = traj_path.replace(".h5", ".json")
json_data = load_json(json_path)

episodes = json_data["episodes"]
ep = episodes[episode_idx]
# episode_id should be the same as episode_idx, unless specified otherwise
episode_id = ep["episode_id"]
traj = h5_file[f"traj_{episode_id}"]
env_states = dict_to_list_of_dicts(traj["env_states"])
#%%
# rename the keys in env_states to match the keys provided by env sim state
for env_state in env_states:
    new_actors_dict = {}
    for actor_key in env_state['actors'].keys():
        if actor_key.endswith("_0"):
            # strip _0 from the actor_key
            new_key = "_".join(actor_key.split("_")[:-1])
            new_actors_dict[new_key] = env_state['actors'][actor_key]
        else:
            new_actors_dict[actor_key] = env_state['actors'][actor_key]
    env_state['actors'] = new_actors_dict

#%%
# Create the environment
env_kwargs = json_data["env_info"]["env_kwargs"]
env = gym.make(json_data["env_info"]["env_id"], **env_kwargs)
print(env_kwargs)
#%%
# Reset the environment
reset_kwargs = ep["reset_kwargs"].copy()
reset_kwargs["seed"] = ep["episode_seed"]
env.reset(**reset_kwargs)
#%%
human_render_cam_params = env.scene.human_render_cameras['render_camera'].get_params()
human_render_cam_intrisic = human_render_cam_params['intrinsic_cv'][0]
human_render_cam_cam2world_gl = human_render_cam_params['cam2world_gl'][0][:3, :4]
human_render_cam_extrinsic_cv = human_render_cam_params['extrinsic_cv'][0]
# #%%
# dummy_point = torch.tensor([0, 0, 0], dtype=torch.float32)
# projected_point = human_render_cam_intrisic @ (human_render_cam_extrinsic_cv @ torch.cat([dummy_point, torch.tensor([1.0])]))
# # projected_point = human_render_cam_intrisic @ (human_render_cam_cam2world_gl @ torch.cat([dummy_point, torch.tensor([1.0])]))
# projected_point = (projected_point[:2] / projected_point[2]).int()
#%%
# render = env.render_rgb_array()[0].numpy()
# plt.imshow(render)
# plt.scatter(projected_point[0], projected_point[1])
#%%
frames = [env.render_rgb_array()[0].numpy()]
for i in tqdm.tqdm(range(len(traj["actions"]))):
    action = traj["actions"][i]
    obs, reward, terminated, truncated, info = env.step(action)
    current_frame = env.render_rgb_array()[0].numpy()
    contact_positions = []
    contacts = env.scene.get_contacts()
    if len(contacts) > 0:
        for contact in contacts:
            for contact_point in contact.points:
                if np.linalg.norm(contact_point.impulse) > 0:
                    contact_positions.append(torch.from_numpy(contact_point.position))
            # contact_positions.extend([torch.from_numpy(contact_point.position) for contact_point in contact.points])
    contact_positions = torch.stack(contact_positions) if len(contact_positions) > 0 else []
    if len(contact_positions) > 0:
        for contact_position in contact_positions:
            projected_point = human_render_cam_intrisic @ (human_render_cam_extrinsic_cv @ torch.cat([contact_position, torch.tensor([1.0])]))
            projected_point = ((projected_point[:2] / projected_point[2]).int()).numpy()
            cv2.circle(current_frame, (projected_point[0], projected_point[1]), 3, (0, 255, 0), -1)
    # env.set_state_dict(env_states[i])
    frames.append(current_frame)
#%%
# contact_positions = [torch.from_numpy(contact_point.position) for contact_point in contacts[0].points]
# contact_positions = torch.stack(contact_positions)
# #%%
# plt.imshow(env.render_rgb_array()[0].numpy())
# for contact_position in contact_positions:
#     projected_point = human_render_cam_intrisic @ (human_render_cam_extrinsic_cv @ torch.cat([contact_position, torch.tensor([1.0])]))
#     projected_point = (projected_point[:2] / projected_point[2]).int()
#     plt.scatter(projected_point[0], projected_point[1], c='red', s=10)

#%%
env.close()
del env
images_to_video(frames, output_dir=".", video_name="replay", fps=30, )
# %%
plt.imshow(env.render_rgb_array()[0].numpy())
# %%
for contact in contacts:
    for contact_point in contact.points:
        print(contact_point.impulse)
# %%
## testing book insertion task
env = gym.make(json_data["env_info"]["env_id"], **env_kwargs)
