# %load_ext autoreload
# %autoreload 2

import os
import pathlib
import sys
import shutil
import argparse
from pathlib import Path

import time
import dask
import pandas as pd
import xarray as xr
import numpy as np
from torch import optim
from torchinfo import summary
from nowproject.dataloader import AutoregressivePatchLearningDataLoader, AutoregressivePatchLearningDataset
from nowproject.models.utils_models import inverse_transform_data_for_raft, reshape_input_for_encoding
from xforecasting.dataloader_autoregressive import get_aligned_ar_batch

from xforecasting.utils.io import get_ar_model_tensor_info
from xforecasting.utils.torch import summarize_model
from xforecasting import (
    AR_Scheduler,
    rechunk_forecasts_for_verification,
    EarlyStopping,
)

from nowproject.config import (
    read_config_file,
    write_config_file,
    get_model_settings,
    get_training_settings,
    get_ar_settings,
    get_dataloader_settings,
    get_pytorch_model,
    get_model_name,
    set_pytorch_settings,
    load_pretrained_model,
    create_experiment_directories,
    print_model_description,
    print_tensor_info,
    create_test_events_time_range
)

from nowproject.verification.verification import verification_routine

from xverif import xverif

# Project specific functions
from torch import nn
import nowproject.architectures as dl_architectures
from nowproject.loss import (
    FSSLoss,
    CombinedFSSLoss, 
    WeightedMSELoss,
    LogCoshLoss, 
    reshape_tensors_4_loss
)
from nowproject.training import AutoregressiveTraining
from nowproject.predictions import AutoregressivePredictions
from nowproject.verification.plot_skills import ( 
    plot_averaged_skill,
    plot_averaged_skills, 
    plot_skills_distribution
)
from nowproject.verification.plot_map import (
    plot_forecast_comparison
)
from nowproject.data.dataset.data_config import METADATA_CH

from nowproject.scalers import (
    Scaler,
    log_normalize_inverse_transform,
    log_normalize_transform,
    normalize_transform,
    normalize_inverse_transform,
    bin_transform,
    bin_inverse_transform
)
from nowproject.data.scalers_modules import (
    log_normalize_scaler,
    normalize_scaler,
    bin_scaler
)
from nowproject.data.dataset.data_config import METADATA, BOTTOM_LEFT_COORDINATES
from nowproject.data.data_utils import (
    load_static_topo_data, 
    prepare_data_dynamic,
    prepare_data_patches, 
    get_tensor_info_with_patches
)

from nowproject.models.layers_res_conv import RAFTOpticalFlow

import torch
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F_vision
from nowproject.models.utils_models import flow_warp
from pysteps.motion.lucaskanade import dense_lucaskanade

default_data_dir = "/ltenas3/0_Data/NowProject/"
default_static_data_path = "/ltenas3/0_GIS/DEM Switzerland/srtm_Switzerland_EPSG21781.tif"
default_exp_dir = "/home/haddad/experiments/"
# default_config = "/home/haddad/nowproject/configs/UNet/AvgPool4-Conv3.json"
default_config = "/home/haddad/nowproject/configs/resConv/conv64_optical_flow.json"
default_test_events = "/home/haddad/nowproject/configs/subset_test_events.json"

cfg_path = Path(default_config)
data_dir_path  = Path(default_data_dir)
static_data_path = Path(default_static_data_path)
test_events_path = Path(default_test_events)
exp_dir_path = Path(default_exp_dir)

t_start = time.time()
cfg = read_config_file(fpath=cfg_path)

##------------------------------------------------------------------------.
# Load experiment-specific configuration settings
model_settings = get_model_settings(cfg)
ar_settings = get_ar_settings(cfg)
training_settings = get_training_settings(cfg)
dataloader_settings = get_dataloader_settings(cfg)

boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                    boundaries=boundaries, 
                                    timestep=5)
# data_static = load_static_topo_data(static_data_path, data_dynamic)
data_static = None

patch_size = 128
data_patches = prepare_data_patches(data_dir_path / "rzc_cropped_patches_fixed.parquet",
                                    patch_size=patch_size,
                                    timestep=5)
data_bc = None

##------------------------------------------------------------------------.
# Load scalers
# scaler = normalize_scaler()
# model_settings["last_layer_activation"] = False

scaler = log_normalize_scaler()
model_settings["last_layer_activation"] = False

# scaler = bin_scaler()
# model_settings["last_layer_activation"] = False
##------------------------------------------------------------------------.
# Split data into train, test and validation set
## - Defining time split for training
# training_years = np.array(["2018-01-01T00:00", "2018-12-31T23:57:30"], dtype="M8[s]")
# validation_years = np.array(["2021-01-01T00:00", "2021-03-31T23:57:30"], dtype="M8[s]")
# training_years = np.array(["2018-01-01T00:00", "2018-06-30T23:57:30"], dtype="M8[s]")
# validation_years = np.array(["2021-01-01T00:00", "2021-01-31T23:57:30"], dtype="M8[s]")
training_years = np.array(["2018-01-01T00:00", "2018-01-03T23:57:30"], dtype="M8[s]")
validation_years = np.array(["2021-01-01T00:00", "2021-01-02T23:57:30"], dtype="M8[s]")
test_events = create_test_events_time_range(test_events_path, freq="5min")

# - Split data sets
t_i = time.time()
training_data_dynamic = data_dynamic.sel(
    time=slice(training_years[0], training_years[-1])
)
training_data_patches = data_patches.sel(
    time=slice(training_years[0], training_years[-1])
)

validation_data_dynamic = data_dynamic.sel(
    time=slice(validation_years[0], validation_years[-1])
)
validation_data_patches = data_patches.sel(
    time=slice(validation_years[0], validation_years[-1])
)

print(
    "- Splitting data into train, validation and test sets: {:.2f}s".format(
        time.time() - t_i
    )
)

##------------------------------------------------------------------------.
# Define pyTorch settings (before PyTorch model definition)
## - Here inside eventually set the seed for fixing model weights initialization
## - Here inside the training precision is set (currently only float32 works)
device = set_pytorch_settings(training_settings)

##------------------------------------------------------------------------.
# Retrieve dimension info of input-output Torch Tensors
tensor_info = get_ar_model_tensor_info(
    ar_settings=ar_settings,
    data_dynamic=training_data_dynamic,
    data_static=data_static,
    data_bc=None,
)

tensor_info = get_tensor_info_with_patches(tensor_info, patch_size)

# - Add dim info to cfg file
model_settings["tensor_info"] = tensor_info
cfg["model_settings"]["tensor_info"] = tensor_info

##------------------------------------------------------------------------.
# Define the model architecture
# model = get_pytorch_model(module=dl_architectures, model_settings=model_settings)

# # If requested, load a pre-trained model for fine-tuning
# if model_settings["pretrained_model_name"] is not None:
#     model_dir = exp_dir_path / model_settings["model_name"]
#     load_pretrained_model(model=model, model_dir=model_dir.as_posix())

# # Transfer model to the device (i.e. GPU)
# model = model.to(device)

# Summarize the model
input_shape = tensor_info["input_shape"]["train"].copy()
input_shape[0] = training_settings["training_batch_size"]
# print(
#     summary(
#         model, input_shape, col_names=["input_size", "output_size", "num_params"]
#     )
# )

# _ = summarize_model(
#     model=model,
#     input_size=tuple(tensor_info["input_shape"]["train"][1:]),
#     batch_size=training_settings["training_batch_size"],
#     device=device,
# )

# Generate the (new) model name and its directories
if model_settings["model_name"] is not None:
    model_name = model_settings["model_name"]
else:
    model_name = get_model_name(cfg)
    model_settings["model_name"] = model_name
    cfg["model_settings"]["model_name_prefix"] = None
    cfg["model_settings"]["model_name_suffix"] = None

model_dir = create_experiment_directories(
    exp_dir=exp_dir_path, model_name=model_name, 
    suffix=f"5mins-Patches-LogNormalizeScaler-MSEMaskedWeighted-{training_settings['epochs']}epochs-3days-test", 
    force=True
)  # force=True will delete existing directory

# Define model weights filepath
model_fpath = model_dir / "model_weights" / "model.h5"

##------------------------------------------------------------------------.
# Write config file in the experiment directory
write_config_file(cfg=cfg, fpath=model_dir / "config.json")


##------------------------------------------------------------------------.
# - Define AR_Weights_Scheduler
## - For RNN: growth and decay works well
if training_settings["ar_training_strategy"] == "RNN":
    initial_ar_absolute_weights = [1] if "Direct" in model_name else [1, 1]
    ar_scheduler = AR_Scheduler(
        method="LinearStep",
        factor=0.0005,
        fixed_ar_weights=[0],
        initial_ar_absolute_weights=initial_ar_absolute_weights,
    )
## - FOR AR : Do not decay weights once they growthed
elif training_settings["ar_training_strategy"] == "AR":
    ar_scheduler = AR_Scheduler(
        method="LinearStep",
        factor=0.0005,
        fixed_ar_weights=np.arange(0, ar_settings["ar_iterations"]),
        initial_ar_absolute_weights=[1, 1],
    )
else:
    raise NotImplementedError(
        "'ar_training_strategy' must be either 'AR' or 'RNN'."
    )



##------------------------------------------------------------------------.
### - Train the model
dask.config.set(
    scheduler="synchronous"
)  # This is very important otherwise the dataloader hang


t_i = time.time()
training_ds = AutoregressivePatchLearningDataset(
    data_dynamic=training_data_dynamic,
    data_bc=None,
    data_patches=None,
    data_static=data_static,
    bc_generator=None,
    scaler=scaler,
    # Custom AR batching function
    ar_batch_fun=get_aligned_ar_batch,
    training_mode=True,
    # Autoregressive settings
    input_k=ar_settings["input_k"],
    output_k=ar_settings["output_k"],
    forecast_cycle=ar_settings["forecast_cycle"],
    ar_iterations=ar_scheduler.current_ar_iterations,
    stack_most_recent_prediction=ar_settings["stack_most_recent_prediction"],
    # Timesteps
    subset_timesteps=(training_data_patches.time.values if training_data_patches is not None else None),
    # GPU settings
    device=device,
)

training_dl = AutoregressivePatchLearningDataLoader(
    dataset=training_ds,
    batch_size=training_settings["training_batch_size"],
    drop_last_batch=dataloader_settings["drop_last_batch"],
    shuffle=dataloader_settings["random_shuffling"],
    shuffle_seed=training_settings["seed_random_shuffling"],
    num_workers=dataloader_settings["num_workers"],
    prefetch_factor=dataloader_settings["prefetch_factor"],
    prefetch_in_gpu=dataloader_settings["prefetch_in_gpu"],
    pin_memory=dataloader_settings["pin_memory"],
    asyncronous_gpu_transfer=dataloader_settings["asyncronous_gpu_transfer"],
    device=device,
)

raft = RAFTOpticalFlow(tensor_info["input_n_time"], small_model=True, finetune=False).to("cuda")

training_iterator = iter(training_dl)
num_batches = len(training_iterator)
batch_indices = range(num_batches)
training_batch_dict = next(training_iterator)
##------------------------------------------------------------.
# Perform autoregressive training loop
# - The number of AR iterations is determined by ar_scheduler.ar_weights
# - If ar_weights are all zero after N forecast iteration:
#   --> Load data just for F forecast iteration
#   --> Autoregress model predictions just N times to save computing time
dict_training_Y_predicted = {}
dict_training_loss_per_ar_iteration = {}
ar_iteration = 0
# Retrieve X and Y for current AR iteration
# - ar_batch_fun() function stack together the required data from the previous AR iteration
torch_X, torch_Y = get_aligned_ar_batch(
    ar_iteration=ar_iteration,
    batch_dict=training_batch_dict,
    dict_Y_predicted=dict_training_Y_predicted,
    asyncronous_gpu_transfer=dataloader_settings["asyncronous_gpu_transfer"],
    device=device,
)


batch_size = torch_X.shape[0]
input_y_dim = tensor_info["input_shape_info"]["test"]["dynamic"]["y"]
input_x_dim = tensor_info["input_shape_info"]["test"]["dynamic"]["x"]

##--------------------------------------------------------------------.
# Reorder and reshape data
x = reshape_input_for_encoding(torch_X, tensor_info["dim_order"]["dynamic"], 
                                [batch_size, 1, tensor_info["input_n_time"], 
                                input_y_dim, input_x_dim])

out, flow_fields, batches = raft(x)

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.imshow(x[0][0][i].cpu(), cmap="Blues", alpha=1)
    ax.imshow(out[0][0][i].cpu().detach().numpy(), cmap="Oranges", alpha=0.4)
plt.show()



fig, axs = plt.subplots(1, 2, figsize=(12, 6))
cmaps = ['Purples', 'Blues', 'Greens', 'Oranges']
axs = axs.flatten()
for i in range(tensor_info["input_n_time"]):
    axs[0].imshow(out[0][0][i].cpu().detach().numpy(), cmap=cmaps[i], alpha=0.3)
    axs[1].imshow(x[0][0][i].cpu(), cmap=cmaps[i], alpha=0.3)

axs[0].imshow(torch_Y[0, 0, :, :, 0].cpu(), cmap="Reds", alpha=0.4)
axs[1].imshow(torch_Y[0, 0, :, :, 0].cpu(), cmap="Reds", alpha=0.4)

plt.show()

pad = (3, 3, 3, 3)

fig, axs = plt.subplots(4, 5, figsize=(24, 24))

for i in range(tensor_info["input_n_time"]):
    flows_to_apply = flow_fields[i:] if i < len(flow_fields) else [flow_fields[-1]]
    batch = batches[i][:1]
    axs[i][i].imshow(inverse_transform_data_for_raft(batch).cpu()[:, 0, pad[2]:-pad[3], pad[0]:-pad[1]][0])
    for j, flow in enumerate(flows_to_apply):
        batch = flow_warp(batch, flow[:1].permute(0, 2, 3, 1),
                            interpolation="nearest", 
                            padding_mode="reflection")
        axs[i][i+j+1].imshow(inverse_transform_data_for_raft(batch).cpu().detach().numpy()[:, 0, pad[2]:-pad[3], pad[0]:-pad[1]][0])
plt.show()


flows = []
for i, flow in enumerate(flow_fields):
    flow_m = flow.clone().cpu().detach().numpy()
    flow_m[:, 0, :, :] = flow_m[:, 0, ::-1, ::-1]
    flow_m[:, 1, :, :] = flow_m[:, 1, ::-1, ::-1]
    flows.append(torch.Tensor(flow_m).to("cuda"))

pad = (3, 3, 3, 3)

cmaps = ["Greys", 'Purples', 'Blues', 'Greens', 'Oranges']
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs = axs.flatten()
for i in range(tensor_info["input_n_time"]):
    flows_to_apply = flows[i:] if i < len(flows) else [flows[-1]]
    batch = batches[i][:1]
    axs[i].imshow(inverse_transform_data_for_raft(batch).cpu()[:, 0, pad[2]:-pad[3], pad[0]:-pad[1]][0], cmap=cmaps[i], alpha=0.4)
    for j, flow in enumerate(flows_to_apply):
        batch = flow_warp(batch, flow[:1].permute(0, 2, 3, 1),
                            interpolation="nearest", 
                            padding_mode="reflection")
        axs[i].imshow(inverse_transform_data_for_raft(batch).cpu().detach().numpy()[:, 0, pad[2]:-pad[3], pad[0]:-pad[1]][0],
                      cmap=cmaps[i+j+1], alpha=0.4)
    
    axs[i].imshow(torch_Y[0, 0, :, :, 0].cpu(), cmap="Reds", alpha=0.4)

plt.show()



subset = x[0][0].cpu().numpy()
flow_lk = torch.Tensor(dense_lucaskanade(subset)).unsqueeze(dim=0)

out_lk = []

cmaps = ["Greys", 'Purples', 'Blues', 'Greens', 'Oranges']
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs = axs.flatten()
for i in range(tensor_info["input_n_time"]):
    batch = torch.stack([x[0, :, i], x[0, :, i], x[0, :, i]], dim=1)\
                 .reshape(1, 3, x.shape[-2], x.shape[-1]).cpu()
    axs[i].imshow(batch[0, 0, :], cmap=cmaps[i], alpha=0.8)
    for j in range(tensor_info["input_n_time"] - i):
        batch = flow_warp(batch, flow_lk.permute(0, 2, 3, 1),
                            interpolation="nearest", 
                            padding_mode="reflection")
        axs[i].imshow(batch[0, 0, :].cpu(), cmap=cmaps[i+j+1], alpha=0.4)
    out_lk.append(batch[:, 0, :])
    axs[i].imshow(torch_Y[0, 0, :, :, 0].cpu(), cmap="Reds", alpha=0.4)


fig, axs = plt.subplots(2, 2, figsize=(16, 12))
cmaps = ['Purples', 'Blues', 'Greens', 'Oranges']
axs = axs.flatten()
for i in range(tensor_info["input_n_time"]):
    axs[0].imshow(out[0][0][i].cpu().detach().numpy(), cmap=cmaps[i], alpha=0.3)
    axs[1].imshow(out_lk[i][0], cmap=cmaps[i], alpha=0.3)
    axs[2].imshow(out[0][0][i].cpu().detach() - out_lk[i][0], cmap=cmaps[i], alpha=0.3)
    axs[3].imshow(x[0][0][i].cpu(), cmap=cmaps[i], alpha=0.3)

axs[0].imshow(torch_Y[0, 0, :, :, 0].cpu(), cmap="Reds", alpha=0.4)
axs[1].imshow(torch_Y[0, 0, :, :, 0].cpu(), cmap="Reds", alpha=0.4)
axs[3].imshow(torch_Y[0, 0, :, :, 0].cpu(), cmap="Reds", alpha=0.4)

axs[0].set_title("RAFT")
axs[1].set_title("Lucas-Kanade")
axs[2].set_title("Difference")
axs[3].set_title("Observations")

plt.show()


flow_imgs = [flow_to_image(flow[0]) for flow in flow_fields]

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()
for i, ax in enumerate(axs):
    img = F_vision.to_pil_image(flow_imgs[i].to("cpu"))
    s = ax.imshow(np.asarray(img))
plt.show()


plt.imshow(flow_fields[0][0][0].cpu().detach().numpy(), cmap="Reds")
# plt.imshow(flow_fields[0][0][0].cpu().detach().numpy(), cmap="Blues")
plt.colorbar()


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F_vision.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()