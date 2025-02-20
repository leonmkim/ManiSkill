#%%
import numpy as np

# %%
rng = np.random.default_rng(0)
val_num_demos = 10
train_num_demos = 20
dummy_demo_idxs_list = np.arange(500)
copy_rng = rng
#%%
rng.shuffle(dummy_demo_idxs_list)
valid_demo_idxs = dummy_demo_idxs_list[:val_num_demos]
train_demo_idxs = np.setdiff1d(dummy_demo_idxs_list, valid_demo_idxs)
copy_rng.shuffle(train_demo_idxs)
train_demo_idxs = train_demo_idxs[:train_num_demos]
print(f"train_demo_idxs: {train_demo_idxs}, valid_demo_idxs: {valid_demo_idxs}")

# %%
