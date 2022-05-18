import pathlib
import xarray as xr
import numpy as np

from nowproject.data.data_config import BOTTOM_LEFT_COORDINATES


def prepare_data_dynamic(data_dynamic_path: pathlib.Path):
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
    data_dynamic = data_dynamic.sel({
        "x": slice(485, 831), 
        "y": slice(301, 75)
    })
    data_dynamic = data_dynamic.rename({"precip": "feature"})[["feature"]]

    return data_dynamic


def load_static_topo_data(topo_data_path: pathlib.Path, data_dynamic: xr.Dataset, 
                          upsample: bool = True, upsample_factor: int = 13) -> xr.Dataset:
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

    Returns
    -------
    xr.Dataset
        Static topographic data with dimensions corresponding to the dynamic data
    """
    dem = xr.open_rasterio(topo_data_path)
    dem = dem.isel(band=0, drop=True)

    if upsample:
        dem = dem.coarsen({"x": upsample_factor, "y": upsample_factor}, boundary="trim").mean()

    dem["x"] = dem["x"] / 1000
    dem["y"] = dem["y"] / 1000
    dem = dem.interp(coords={"x": data_dynamic.x.values, "y": data_dynamic.y.values})
    

    return dem.to_dataset(name="feature")