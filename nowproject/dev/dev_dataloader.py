from pathlib import Path

import time
import dask
import pandas as pd
import xarray as xr
import numpy as np
from torch import optim
from torchinfo import summary

from xforecasting.utils.io import get_ar_model_tensor_info
from xforecasting import (
    AR_Scheduler,
    AutoregressivePredictions,
    rechunk_forecasts_for_verification,
    EarlyStopping,
)

from nowproject.utils.config import (
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

from nowproject.utils.scalers import RainScaler, RainBinScaler
from nowproject.data.data_utils import load_static_topo_data, prepare_data_dynamic
from nowproject.dataloader import AutoregressivePatchLearningDataLoader, AutoregressivePatchLearningDataset, autoregressive_patch_collate_fn
from xforecasting.dataloader_autoregressive import get_aligned_ar_batch

# %load_ext autoreload
# %autoreload 2


default_data_dir = "/ltenas3/0_Data/NowProject/"
default_static_data_path = "/ltenas3/0_GIS/DEM Switzerland/srtm_Switzerland_EPSG21781.tif"
default_exp_dir = "/home/haddad/experiments/"
default_config = "/home/haddad/nowproject/configs/UNet/AvgPool4-Conv3.json"
default_test_events = "/home/haddad/nowproject/configs/events.json"

cfg_path = Path(default_config)
data_dir_path  = Path(default_data_dir)
static_data_path = Path(default_static_data_path)
test_events_path = Path(default_test_events)
exp_dir_path = Path(default_exp_dir)


cfg = read_config_file(fpath=cfg_path)
##------------------------------------------------------------------------.
# Load experiment-specific configuration settings

model_settings = get_model_settings(cfg)
ar_settings = get_ar_settings(cfg)
training_settings = get_training_settings(cfg)
dataloader_settings = get_dataloader_settings(cfg)
##------------------------------------------------------------------------.
# Load Zarr Datasets

boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                    boundaries=boundaries)

data_static = load_static_topo_data(static_data_path, data_dynamic)
data_patches = pd.read_parquet(data_dir_path / "rzc_cropped_patches.parquet")
data_patches = data_patches.groupby("time")["upper_left_idx"].apply(lambda x: ', '.join(x)).to_xarray()
data_patches = data_patches.assign_attrs({"patch_size": 128})
data_patches = data_patches.reindex({"time": data_dynamic.time.values}, fill_value="")
data_bc = None
##------------------------------------------------------------------------.
# Load scalers

scaler = RainScaler(feature_min=np.log10(0.025), 
                    feature_max=np.log10(150), 
                    threshold=np.log10(0.1))


training_years = np.array(["2018-10-01T00:00", "2018-10-31T23:57:30"], dtype="M8[s]")
validation_years = np.array(["2021-01-01T00:00", "2021-01-10T23:57:30"], dtype="M8[s]")
test_events = create_test_events_time_range(test_events_path)[:1]
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


device = set_pytorch_settings(training_settings)
##------------------------------------------------------------------------.
# Retrieve dimension info of input-output Torch Tensors

tensor_info = get_ar_model_tensor_info(
    ar_settings=ar_settings,
    data_dynamic=training_data_dynamic,
    data_static=data_static,
    data_bc=None,
)

print_tensor_info(tensor_info)
# - Add dim info to cfg file

model_settings["tensor_info"] = tensor_info
cfg["model_settings"]["tensor_info"] = tensor_info

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

training_ds = AutoregressivePatchLearningDataset(
    data_dynamic=training_data_dynamic,
    data_bc=None,
    data_static=data_static,
    data_patches=training_data_patches,
    bc_generator=None,
    scaler=scaler,
    # Custom AR batching function
    ar_batch_fun=get_aligned_ar_batch,
    training_mode=True,
    # Autoregressive settings
    input_k=ar_settings["input_k"],
    output_k=ar_settings["output_k"],
    forecast_cycle=ar_settings["forecast_cycle"],
    ar_iterations=ar_settings["ar_iterations"],
    stack_most_recent_prediction=ar_settings["stack_most_recent_prediction"],
    # GPU settings
    device=device,
)


list_samples = [training_ds.__getitem__(i) for i in range(16)]

# [list_samples[1]["X_dynamic"][idx].shape for idx in list_samples[1]["X_dynamic"] if list_samples[0]["X_dynamic"][idx] is not None]

batch_size = 16

torch_static = training_ds.torch_static
dim_info = training_ds.dim_info["static"]
batch_dim = dim_info["sample"]
new_dim_size = [-1 for i in range(torch_static.dim())]
new_dim_size[batch_dim] = batch_size
torch_static = torch_static.expand(new_dim_size)


batch_dict = autoregressive_patch_collate_fn(list_samples, 
                                torch_static=torch_static, 
                                prefetch_in_gpu=dataloader_settings["prefetch_in_gpu"],
                                pin_memory=dataloader_settings["pin_memory"],
                                device=device)

