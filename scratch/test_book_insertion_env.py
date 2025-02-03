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
# %%
## testing book insertion task
env = gym.make("BookInsertion-v0", reward_mode="none", sim_backend='physx_cuda')

# %%
env.reset()
#%%
frame = env.render_rgb_array()[0].cpu().numpy()
plt.imshow(frame)
#%%
frames = [env.render_rgb_array()[0].cpu().numpy()]
for i in tqdm.tqdm(range(50)):
    action = env.action_space.sample()
    # grasp the book
    action[-1]= -1
    obs, reward, terminated, truncated, info = env.step(action)
    current_frame = env.render_rgb_array()[0].cpu().numpy()
    frames.append(current_frame)
# %%
images_to_video(frames, output_dir=".", video_name="book_insertion", fps=30, )

# %%
