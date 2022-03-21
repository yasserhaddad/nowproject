import pathlib
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pysteps.visualization.precipfields import plot_precip_field
from nowproject.data.data_config import METADATA

zarr_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/zarr_test/")

ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")
ds = ds.rename({"precip": "feature"})

ax = plot_precip_field(ds.isel({"time":0}).precip, geodata=METADATA)
ax.add_feature(cfeature.BORDERS)
plt.plot()