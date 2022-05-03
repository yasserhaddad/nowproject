import os
import sys
import shutil
import argparse
from pathlib import Path

import time
import dask
import xarray as xr
import numpy as np
from torch import optim
from torchinfo import summary

from xforecasting.utils.io import get_ar_model_tensor_info
from xforecasting.utils.torch import summarize_model
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
    create_test_events_autoregressive_time_range
)

from xverif import xverif

# Project specific functions
import nowproject.architectures as dl_architectures
from nowproject.loss import WeightedMSELoss, reshape_tensors_4_loss
from nowproject.training import AutoregressiveTraining

model_dir = Path("/home/haddad/experiments/RNN-AR6-UNet-AvgPooling/")


zarr_dir_path = Path("/ltenas3/0_Data/NowProject/zarr/")

ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")

ds = ds.sel({"y": list(range(850, 450, -1)), "x": list(range(30, 320))})

ds = ds.rename({"precip": "feature"})

forecast_zarr_fpath = (
        model_dir / "model_predictions" / "forecast_chunked" / "test_forecasts.zarr"
    )

ds_forecasts = xr.open_zarr(forecast_zarr_fpath)
print(ds_forecasts)
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
    model_dir / "model_predictions" / "space_chunked" / "test_forecasts_2.zarr"
).as_posix()

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
    chunks="auto",
    target_store=verification_zarr_fpath,
    max_mem="30GB",
)
print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i) / 60))

verification_zarr_fpath = (
    model_dir / "model_predictions" / "space_chunked" / "test_forecasts.zarr"
).as_posix()


ds_verification_format = xr.open_dataset(verification_zarr_fpath).load()

selection = ds.sel(time=ds_verification_format.time).load()

ds_skill = xverif.deterministic(
        pred=ds_verification_format.chunk({"x": 1, "y": 1}),
        obs=selection.chunk({"x": 1, "y": 1}),
        forecast_type="continuous",
        aggregating_dim="time",
    )

ds_skill.to_netcdf((model_dir / "model_skills" / "deterministic_spatial_skill.nc"))

ds_skill = xr.open_dataset(model_dir / "model_skills" / "deterministic_spatial_skill.nc")

ds_averaged_skill = xr.open_dataset(model_dir / "model_skills" / "deterministic_global_skill.nc")
