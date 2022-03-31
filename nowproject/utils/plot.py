import pathlib
import xarray as xr

import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pysteps.visualization.precipfields import plot_precip_field
from nowproject.data.data_config import METADATA

zarr_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/zarr/")

ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")
# ds = ds.rename({"precip": "feature"})

crs = pyproj.CRS.from_epsg(21781)

METADATA = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 255000.0,
    "y1": -160000.0,
    "x2": 894000.0,
    "y2": 549000.0,
    "xpixelsize": 1000.0,
    "ypixelsize": 1000.0,
    "cartesian_unit": "m",
    "yorigin": "upper",
    "institution": "MeteoSwiss",
    "product": "RZC",
    "accutime": 2.5,
    "unit": 'mm/h',
    "zr_a": 316.0,
    "zr_b": 1.5
}

bbox = (451000, 30000, 850000, 321000)

ax = plot_precip_field(ds.isel({"time":0}).precip, geodata=METADATA)
# ax.add_feature(cfeature.BORDERS)
plt.show()

ds_masked = ds.sel({"y": list(range(850, 450, -1)), "x": list(range(30, 320))})


from pyproj import Transformer

transformer = Transformer.from_crs(crs, crs.geodetic_crs)


import cartopy
import cartopy.feature as cfeature
# # Set Swiss 
# proj_crs = cartopy.crs.epsg(21781)
# fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))

# fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection=ccrs.PlateCarree())
lat_lon = transformer.transform(450000, 30000)
print(lat_lon)
lat_lon_2 = transformer.transform(850000, 320000)
print(lat_lon_2)
ax.set_extent([lat_lon[1], lat_lon[0], lat_lon_2[1], lat_lon_2[0]], crs=ccrs.PlateCarree())

# Add borders
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)

# # Add gridlines 
# gl = ax.gridlines(draw_labels=True, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False 
plt.show()
	
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
ax.add_feature(cartopy.feature.LAKES, alpha=0.95)
ax.add_feature(cartopy.feature.RIVERS)
 
# ax.set_extent([-50, 60, -25, 60])
ax.set_extent([lat_lon[1], lat_lon[0], lat_lon_2[1], lat_lon_2[0]])
plt.show()
