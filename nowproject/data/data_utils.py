import copy
import pathlib
import pandas as pd
import xarray as xr
import numpy as np

from topo_descriptors import topo, helpers
from nowproject.data.data_config import BOTTOM_LEFT_COORDINATES

def xr_sel_coords_between(data, **kwargs):
    for k, slc in kwargs.items():
        if not isinstance(slc, slice):
             raise TypeError("Expects slice objects.")
        # TODO check coord is 1 dim and exist 
        if k not in data.dims:
            raise ValueError("Argument should be a dimension of the data object.")
        if len(data[k].shape) > 1:
            raise ValueError("Dimension should be 1-D.")
        if slc.stop >= slc.start:
            data = data.sel({k: slc})
        else:
            isel_idx = np.where(np.logical_and(data[k].data > slc.stop, data[k].data <= slc.start))[0]
            data = data.isel({k: isel_idx})
    return data 


def prepare_data_dynamic(data_dynamic_path: pathlib.Path, boundaries: dict = None, 
                         flip: bool = True, timestep: int = None) -> xr.Dataset:
    """Prepare dynamic precipitation data, by adjusting the coordinates, cropping the data
    and select a timestep.

    Parameters
    ----------
    data_dynamic_path : pathlib.Path
        Path to the dynamic data to load
    boundaries : dict, optional
        Boundaries to crop the data, by default None
    flip : bool, optional
        If true, flip the data along the y-axis, by default True
    timestep : int, optional
        If specified, time spacing between each value of the data, by default None

    Returns
    -------
    xr.Dataset
        Preprocessed dynamic precipitation data
    """
    data_dynamic = xr.open_zarr(data_dynamic_path)
    rzc_shape = data_dynamic.isel(time=0).precip.data.shape
    x = np.arange(BOTTOM_LEFT_COORDINATES[0], BOTTOM_LEFT_COORDINATES[0] + rzc_shape[1]) + 0.5
    y = np.arange(BOTTOM_LEFT_COORDINATES[1] + rzc_shape[0] - 1, BOTTOM_LEFT_COORDINATES[1] - 1, -1) + 0.5
    data_dynamic = data_dynamic.assign_coords({"x": x, "y": y})
    data_dynamic = data_dynamic.reset_coords(
        ["radar_names", "radar_quality", "radar_availability"], 
        drop=True
        )
    data_dynamic = data_dynamic.sel(time=slice(None, "2021-09-01T00:00"))
    if timestep:
        t = pd.date_range(start=data_dynamic.time.values[0], 
                            end=data_dynamic.time.values[-1], 
                            freq=f"{timestep}min")
        t = np.intersect1d(t, data_dynamic.time.values)
        data_dynamic = data_dynamic.sel(time=t)
        # data_dynamic = data_dynamic.sel(time=(data_dynamic.time.dt.minute % timestep == 0))
    
    data_dynamic = data_dynamic.rename({"precip": "feature"})[["feature"]]
    data_dynamic = data_dynamic.where(((data_dynamic > 0.04) | data_dynamic.isnull()), 0.0)
    if flip:
        data_dynamic.feature.data = data_dynamic.feature.data[:, ::-1, :]
    if boundaries:
        data_dynamic = xr_sel_coords_between(data_dynamic, **boundaries)
    return data_dynamic


def prepare_data_patches(data_patches_path: pathlib.Path, patch_size: int, timestep: int = None) -> xr.DataArray:
    data_patches = pd.read_parquet(data_patches_path)
    data_patches = data_patches.groupby("time")["upper_left_idx"].apply(lambda x: ', '.join(x)).to_xarray()
    if timestep:
        t = pd.date_range(start=data_patches.time.values[0], 
                            end=data_patches.time.values[-1], 
                            freq=f"{timestep}min")
        t = np.intersect1d(t, data_patches.time.values)
        data_patches = data_patches.sel(time=t)
        # data_patches = data_patches[data_patches.time.dt.minute % timestep == 0]
    data_patches = data_patches.assign_attrs({"patch_size": patch_size})

    return data_patches


def load_static_topo_data(topo_data_path: pathlib.Path, data_dynamic: xr.Dataset, 
                          upsample: bool = True, upsample_factor: int = 13, 
                          tpi: bool = False, scale_tpi: int = None) -> xr.Dataset:
    """Loads topographic data and resamples it to match the dynamic data loaded for training.

    Parameters
    ----------
    topo_data_path : pathlib.Path
        Path to topographic data file
    data_dynamic : xr.Dataset
        Dynamic data used for training
    upsample : bool
        If true, upsample the topographic data by the upsample_factor
    upsample_factor : int
        Upsampling factor
    tpi : bool
        If true, compute the topographical position index
    scale_tpi : int
        Scale in meters to compute the topographical position index

    Returns
    -------
    xr.Dataset
        Static topographic data with dimensions corresponding to the dynamic data
    """
    dem = xr.open_rasterio(topo_data_path)
    dem = dem.isel(band=0, drop=True)

    if upsample and not tpi:
        dem = dem.coarsen({"x": upsample_factor, "y": upsample_factor}, boundary="trim").mean()
    
    if tpi and not upsample:
        if not scale_tpi:
            raise ValueError("To compute the tpi, please specify the scale_tpi argument.")
        
        scale_pixel, __ = helpers.scale_to_pixel(scale_tpi, dem)
        dem = topo.tpi(dem, scale_pixel)

    dem["x"] = dem["x"] / 1000
    dem["y"] = dem["y"] / 1000
    dem = dem.interp(coords={"x": data_dynamic.x.values, "y": data_dynamic.y.values})

    return dem.to_dataset(name="feature")


def get_tensor_info_with_patches(tensor_info, patch_size=None):
    """Modify the tensor info dictionary to include patch size, or not,
    in the training phase

    Parameters
    ----------
    tensor_info : dict
        Tensor information dictionary
    patch_size : int, optional
        Size of the patches to pass for training, by default None

    Returns
    -------
    dict
        Modified tensor info, with a train and test entry for both dynamic
    and static data.
    """
    test_input_info = copy.deepcopy(tensor_info["input_shape_info"])
    test_output_info = copy.deepcopy(tensor_info["output_shape_info"])

    train_input_info = copy.deepcopy(tensor_info["input_shape_info"])
    train_input_info["dynamic"]["y"] = patch_size if patch_size else train_input_info["dynamic"]["y"]
    train_input_info["dynamic"]["x"] = patch_size if patch_size else train_input_info["dynamic"]["x"]
    if train_input_info["static"]:
        train_input_info["static"]["y"] = patch_size if patch_size else train_input_info["static"]["y"]
        train_input_info["static"]["x"] = patch_size if patch_size else train_input_info["static"]["x"]

    train_output_info = copy.deepcopy(tensor_info["output_shape_info"])
    train_output_info["dynamic"]["y"] = patch_size if patch_size else train_output_info["dynamic"]["y"]
    train_output_info["dynamic"]["x"] = patch_size if patch_size else train_output_info["dynamic"]["x"]

    tensor_info["input_shape_info"] = {
        "train": train_input_info,
        "test": test_input_info,
    }

    tensor_info["output_shape_info"] = {
        "train": train_output_info,
        "test": test_output_info,
    }
    test_tensor_input_shape = copy.deepcopy(tensor_info["input_shape"])
    test_tensor_output_shape = copy.deepcopy(tensor_info["output_shape"])


    tensor_info["input_shape"] = {
        "train": [
                v if k != "feature" else tensor_info["input_n_feature"]
                for k, v in tensor_info["input_shape_info"]["train"]["dynamic"].items()
            ],
        "test": test_tensor_input_shape
    }

    tensor_info["output_shape"] = {
        "train": [
                v if k != "feature" else tensor_info["output_n_feature"]
                for k, v in tensor_info["output_shape_info"]["train"]["dynamic"].items()
            ],
        "test": test_tensor_output_shape
    }

    return tensor_info