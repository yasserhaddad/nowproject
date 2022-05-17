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
from nowproject.data.data_utils import prepare_data_dynamic
import nowproject.architectures as dl_architectures
from nowproject.loss import WeightedMSELoss, reshape_tensors_4_loss
from nowproject.training import AutoregressiveTraining
from nowproject.utils.scalers import RainBinScaler, RainScaler
from nowproject.utils.plot import plot_averaged_skill, plot_averaged_skills, plot_skills_distribution

model_dir = Path("/home/haddad/experiments/RNN-AR6-UNet-AvgPooling/")

# %load_ext autoreload
# %autoreload 2


# zarr_dir_path = Path("/ltenas3/0_Data/NowProject/zarr/")

# ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")

# ds = ds.sel({"y": list(range(850, 450, -1)), "x": list(range(30, 320))})

# ds = ds.rename({"precip": "feature"})

data_dir_path = Path("/ltenas3/0_Data/NowProject/")

data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr")

test = data_dynamic.isel(time=slice(0, 100))
scaler = RainScaler(feature_min=np.log10(0.025), feature_max=np.log10(100), threshold=np.log10(0.1))


test_bins = data_dynamic.isel(time=slice(1300, 1800)).copy()
np_bins = np.ones(test_bins.feature.shape)

np_bins[:10, 150:250, 200:300] = 3
np_bins[20:30, 150:250, 200:300] = 9
np_bins[40:60, 150:250, 200:300] = 14
np_bins[80:90, 150:250, 200:300] = np.nan
np_bins[90:100, 150:250, 200:300] = 500
test_bins["feature"] = (test_bins.feature.dims, np_bins)

bins = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0, 50.0, 80.0, 120.0, 250.0]
# bins = np.logspace(start=np.log10(0.001), stop=np.log10(200), num=10)
centres = [0] + [(bins[i] + bins[i+1])/2 for i in range(0, len(bins)-1)] + [np.nan]
scaler = RainBinScaler(bins, centres)

test_transformed = scaler.transform(test, variable_dim="feature")
test_untransformed = scaler.inverse_transform(test_transformed, variable_dim="feature")


dem_dir_path = Path("/ltenas3/0_GIS/DEM Switzerland/")
dem = xr.open_rasterio(dem_dir_path / "srtm_Switzerland_EPSG21781.tif")
dem = dem.isel(band=0, drop=True)
dem = dem.rename({"x": "y", "y": "x"})
new_y = [y*1000 for y in data_dynamic.y.values[::-1]]
new_x = [x*1000 for x in data_dynamic.x.values]
dem = dem.interp(coords={"x": new_x, "y": new_y})
dem["x"] = dem["x"] / 1000
dem["y"] = dem["y"] / 1000
dem = dem.reindex(y=list(reversed(dem.y))).transpose("y", "x")

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

rechunked_zarr_fpath = (
    model_dir / "model_predictions" / "space_chunked" / "rechunked_store.zarr"
)

ds_rechunked = xr.open_zarr(rechunked_zarr_fpath, chunks="auto")


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


ds_verification_format = xr.open_zarr(verification_zarr_fpath).load()

selection = data_dynamic.sel(time=ds_verification_format.time).load()


####################################################
# Skills

ds_skill = xverif.deterministic(
        pred=ds_verification_format.chunk({"x": 1, "y": 1}),
        obs=selection.chunk({"x": 1, "y": 1}),
        forecast_type="continuous",
        aggregating_dim="time",
    )

ds_skill.to_netcdf((model_dir / "model_skills" / "deterministic_spatial_skill.nc"))

ds_skill = xr.open_dataset(model_dir / "model_skills" / "deterministic_spatial_skill.nc")

ds_averaged_skill = xr.open_dataset(model_dir / "model_skills" / "deterministic_global_skill.nc")

plot_averaged_skill(ds_averaged_skill, skill="RMSE", variables=["feature"]).savefig(
        model_dir / "figs" / "skills" / "RMSE_skill.png"
    )

plot_averaged_skills(ds_averaged_skill, variables=["feature"]).savefig(
        model_dir / "figs" / "skills" / "averaged_skill.png"
    )
plot_skills_distribution(ds_skill, variables=["feature"]).savefig(
    model_dir / "figs" / "skills" / "skills_distribution.png",
)