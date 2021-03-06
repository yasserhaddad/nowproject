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
from nowproject.utils.utils_models import inverse_transform_data_for_raft, reshape_input_for_encoding
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

from nowproject.utils.verification import verification_routine

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
from nowproject.utils.plot_skills import ( 
    plot_averaged_skill,
    plot_averaged_skills, 
    plot_skills_distribution
)
from nowproject.utils.scalers import RainScaler, RainBinScaler
from nowproject.data.dataset.data_config import METADATA, BOTTOM_LEFT_COORDINATES
from nowproject.data.data_utils import load_static_topo_data, prepare_data_dynamic, get_tensor_info_with_patches

# %load_ext autoreload
# %autoreload 2


default_data_dir = "/ltenas3/0_Data/NowProject/"
default_static_data_path = "/ltenas3/0_GIS/DEM Switzerland/srtm_Switzerland_EPSG21781.tif"
default_exp_dir = "/home/haddad/experiments/"
# default_config = "/home/haddad/nowproject/configs/UNet/AvgPool4-Conv3.json"
default_config = "/home/haddad/nowproject/configs/resConv/conv64.json"
default_test_events = "/home/haddad/nowproject/configs/subset_test_events.json"

cfg_path = Path(default_config)
data_dir_path  = Path(default_data_dir)
static_data_path = Path(default_static_data_path)
test_events_path = Path(default_test_events)
exp_dir_path = Path(default_exp_dir)
model_dir = Path("/home/haddad/experiments/RNN-AR6-resConv64-IncrementLearning-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year")

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
training_years = np.array(["2018-01-01T00:00", "2018-12-31T23:57:30"], dtype="M8[s]")
validation_years = np.array(["2021-01-01T00:00", "2021-03-31T23:57:30"], dtype="M8[s]")
# training_years = np.array(["2018-01-01T00:00", "2018-06-30T23:57:30"], dtype="M8[s]")
# validation_years = np.array(["2021-01-01T00:00", "2021-01-31T23:57:30"], dtype="M8[s]")
# training_years = np.array(["2018-01-01T00:00", "2018-01-03T23:57:30"], dtype="M8[s]")
# validation_years = np.array(["2021-01-01T00:00", "2021-01-02T23:57:30"], dtype="M8[s]")
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
# Print model settings
print_model_description(cfg)

##------------------------------------------------------------------------.
# Define the model architecture
model = get_pytorch_model(module=dl_architectures, model_settings=model_settings)

# Transfer model to the device (i.e. GPU)
model = model.to(device)

# Summarize the model
input_shape = tensor_info["input_shape"]["train"].copy()
input_shape[0] = training_settings["training_batch_size"]
print(
    summary(
        model, input_shape, col_names=["input_size", "output_size", "num_params"]
    )
)

_ = summarize_model(
    model=model,
    input_size=tuple(tensor_info["input_shape"]["train"][1:]),
    batch_size=training_settings["training_batch_size"],
    device=device,
)

# Generate the (new) model name and its directories
if model_settings["model_name"] is not None:
    model_name = model_settings["model_name"]
else:
    model_name = get_model_name(cfg)
    model_settings["model_name"] = model_name
    cfg["model_settings"]["model_name_prefix"] = None
    cfg["model_settings"]["model_name_suffix"] = None


# Define model weights filepath
model_fpath = model_dir / "model_weights" / "model.h5"
load_pretrained_model(model=model, model_dir=model_dir.as_posix())
##------------------------------------------------------------------------.
# Write config file in the experiment directory
write_config_file(cfg=cfg, fpath=model_dir / "config.json")

##------------------------------------------------------------------------.
# - Define custom loss function
criterion = WeightedMSELoss(reduction="mean_masked")
# criterion = WeightedMSELoss(reduction="mean_masked", zero_value=1)
# criterion = WeightedMSELoss(reduction="mean_masked",
#                             weighted_truth=True, weights_params=(5, 4))
# criterion = LogCoshLoss(masked=True, weighted_truth=True, weights_params=(5, 4))
# criterion = FSSLoss(mask_size=3)
# criterion = CombinedFSSLoss(mask_size=3, cutoffs=[0.5, 5.0, 10.0])
##------------------------------------------------------------------------.
# - Define optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=training_settings["learning_rate"],
    eps=1e-7,
    weight_decay=0,
    amsgrad=False,
)

##------------------------------------------------------------------------.
# - Define AR_Weights_Scheduler
## - For RNN: growth and decay works well
if training_settings["ar_training_strategy"] == "RNN":
    ar_scheduler = AR_Scheduler(
        method="LinearStep",
        factor=0.0005,
        fixed_ar_weights=[0],
        initial_ar_absolute_weights=[1, 1],
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
# - Define Early Stopping
## - Used also to update ar_scheduler (aka increase AR iterations) if 'ar_iterations' not reached.
patience = int(
    3000 / training_settings["scoring_interval"]
)  # with 1000 and lr 0.005 crashed without AR update !
minimum_iterations = 20000  # wtih 8000 worked
minimum_improvement = 0.0001
stopping_metric = "validation_total_loss"  # training_total_loss
mode = "min"  # MSE best when low
early_stopping = EarlyStopping(
    patience=patience,
    minimum_improvement=minimum_improvement,
    minimum_iterations=minimum_iterations,
    stopping_metric=stopping_metric,
    mode=mode,
)

##------------------------------------------------------------------------.
### - Define LR_Scheduler
lr_scheduler = None

##-------------------------------------------------------------------------.
### - Create predictions
print("========================================================================================")
print("- Running predictions")
forecast_zarr_fpath = (
        model_dir / "model_predictions" / "forecast_chunked" / "test_forecasts.zarr"
    )
if forecast_zarr_fpath.exists():
    shutil.rmtree(forecast_zarr_fpath)

dask.config.set(scheduler="synchronous")  # This is very important otherwise the dataloader hang 

ds_forecasts = AutoregressivePredictions(
    model=model,
    forecast_reference_times=np.concatenate(test_events), 
    # Data
    data_dynamic=data_dynamic,
    data_static=data_static,
    data_bc=None,
    scaler_transform=scaler,
    scaler_inverse=scaler,
    # Dataloader options
    device=device,
    batch_size=10,  # number of forecasts per batch
    num_workers=dataloader_settings["num_workers"],
    prefetch_factor=dataloader_settings["prefetch_factor"],
    prefetch_in_gpu=dataloader_settings["prefetch_in_gpu"],
    pin_memory=dataloader_settings["pin_memory"],
    asyncronous_gpu_transfer=dataloader_settings["asyncronous_gpu_transfer"],
    # Autoregressive settings
    input_k=ar_settings["input_k"],
    output_k=ar_settings["output_k"],
    forecast_cycle=ar_settings["forecast_cycle"],
    stack_most_recent_prediction=ar_settings["stack_most_recent_prediction"],
    ar_iterations=11,  # How many time to autoregressive iterate
    # Save options
    zarr_fpath=forecast_zarr_fpath.as_posix(),  # None --> do not write to disk
    rounding=2,  # Default None. Accept also a dictionary
    compressor="auto",  # Accept also a dictionary per variable
    chunks="auto",
)

ds_forecasts = xr.open_zarr(forecast_zarr_fpath)
##------------------------------------------------------------------------.
### Reshape forecast Dataset for verification
# - For efficient verification, data must be contiguous in time, but chunked over space (and leadtime)
# - It also neeed to swap from 'forecast_reference_time' to the (forecasted) 'time' dimension
#   The (forecasted) 'time'dimension is calculed as the 'forecast_reference_time'+'leadtime'
print("========================================================================================")
print("- Rechunk and reshape test set forecasts for verification")
dask.config.set(scheduler="threads")
t_i = time.time()
verification_zarr_fpath = (
    model_dir / "model_predictions" / "space_chunked" / "test_forecasts.zarr"
).as_posix()

if "time" in list(ds_forecasts.dims.keys()):
    ds_forecasts = ds_forecasts.drop_dims("time")

# Check the chunk size of coords. If chunk size > coord shape, chunk size = coord shape.
for coord in ds_forecasts.coords:
    if "chunks" in ds_forecasts[coord].encoding:
        new_chunks = []
        for i, c in enumerate(ds_forecasts[coord].encoding["chunks"]):
            if c >= ds_forecasts[coord].shape[i]:
                new_chunks.append(ds_forecasts[coord].shape[i])
            else:
                new_chunks.append(c)
        ds_forecasts[coord].encoding["chunks"] = tuple(new_chunks)


ds_verification_format = rechunk_forecasts_for_verification(
    ds=ds_forecasts,
    chunks={'forecast_reference_time': -1, 'leadtime': 1, "x": 30, "y": 30},
    target_store=verification_zarr_fpath,
    max_mem="30GB",
)
print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i) / 60))
##------------------------------------------------------------------------.
### - Run deterministic verification
print("========================================================================================")
print("- Run deterministic verification")
# dask.config.set(scheduler='processes')
# - Compute skills
ds_det_cont_skill, ds_det_cat_skills, ds_det_spatial_skills, \
        ds_cont_averaged_skill, ds_cat_averaged_skills, ds_spatial_averaged_skills = \
        verification_routine(ds_verification_format, data_dynamic, model_dir / "model_skills")


##------------------------------------------------------------------------.
### - Create verification summary plots and maps
print("========================================================================================")
print("- Create verification summary plots and maps")

# - Create skill vs. leadtime plots
plot_averaged_skill(ds_cont_averaged_skill, skill="RMSE", variables=["feature"]).savefig(
    model_dir / "figs" / "skills" / "RMSE_skill.png"
)
plot_averaged_skills(ds_cont_averaged_skill, variables=["feature"]).savefig(
    model_dir / "figs" / "skills" / "averaged_continuous_skill.png"
)

plot_skills_distribution(ds_det_cont_skill, variables=["feature"]).savefig(
    model_dir / "figs" / "skills" / "skills_distribution.png",
)

for thr in ds_cat_averaged_skills:
    plot_averaged_skills(ds_cat_averaged_skills[thr], 
                    skills=["POD", "FAR", "FA", "ACC", "CSI", "F1"], variables=["feature"]).savefig(
        model_dir / "figs" / "skills" / f"averaged_categorical_skills_thr{thr}.png"
    )
    for spatial_scale in ds_spatial_averaged_skills[thr]:
        plot_averaged_skills(ds_spatial_averaged_skills[thr][spatial_scale], 
                        skills=["SSIM", "FSS"], variables=["feature"], figsize=(12, 8)).savefig(
            model_dir / "figs" / "skills" / f"averaged_spatial_skills_thr{thr}_scale{spatial_scale}.png"
        )

##------------------------------------------------------------------------.
### - Create animations for some forecast 
print("========================================================================================")
print("- Create forecast animations")
t_i = time.time()
timesteps = [
    "2016-04-16 18:00:00",
    "2017-01-12 17:00:00",
    "2017-01-31 16:00:00",
    "2017-06-14 16:00:00",
    "2017-07-07 16:00:00",
    "2017-08-31 17:00:00"
]
for timestep in timesteps:
    ds_forecast_event = ds_forecasts.sel(forecast_reference_time=np.datetime64(timestep))
    plot_forecast_comparison(model_dir / "figs" / "forecast_animations", 
                                ds_forecast_event,
                                data_dynamic,
                                geodata=METADATA_CH,
                                suptitle_prefix=f"{model_settings['architecture_name']}, Increment Learning, ")

print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i) / 60))
##-------------------------------------------------------------------------.
print("========================================================================================")
print(
    "- Model training and verification terminated. Elapsed time: {:.1f} hours ".format(
        (time.time() - t_start) / 60 / 60
    )
)
print("========================================================================================")
##-------------------------------------------------------------------------.
