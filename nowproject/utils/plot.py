import pathlib
from typing import List, Tuple
import numpy as np
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

ax = plot_precip_field(ds.isel({"time":0}).precip, geodata=METADATA, bbox=bbox)
# ax.add_feature(cfeature.BORDERS)
plt.show()

ds_masked = ds.sel({"y": list(range(850, 450, -1)), "x": list(range(30, 320))})


from xarray.plot.utils import _add_colorbar, label_from_attrs
from pysteps.visualization.utils import get_geogrid, get_basemap_axis




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


SKILL_CMAP_DICT = {
    "error_CoV": plt.get_cmap("RdYlBu"),
    "obs_CoV": plt.get_cmap("YlOrRd"),
    "pred_CoV": plt.get_cmap("YlOrRd"),
    # Magnitude
    "BIAS": plt.get_cmap("BrBG"),
    "relBIAS": plt.get_cmap("BrBG"),
    "percBIAS": plt.get_cmap("BrBG"),
    "MAE": plt.get_cmap("Reds"),
    "relMAE": plt.get_cmap("Reds"),
    "percMAE": plt.get_cmap("Reds"),
    "MSE": plt.get_cmap("Reds"),
    "relMSE": plt.get_cmap("Reds"),
    "RMSE": plt.get_cmap("Reds"),
    "relRMSE": plt.get_cmap("Reds"),
    # Average
    "rMean": plt.get_cmap("BrBG"),
    "diffMean": plt.get_cmap("BrBG"),
    # Variability
    "rSD": plt.get_cmap("PRGn"),
    "diffSD": plt.get_cmap("PRGn"),
    "rCoV": plt.get_cmap("PRGn"),
    "diffCoV": plt.get_cmap("PRGn"),
    # Correlation
    "pearson_R": plt.get_cmap("Greens"),
    "pearson_R2": plt.get_cmap("Greens"),
    "spearman_R": plt.get_cmap("Greens"),
    "spearman_R2": plt.get_cmap("Greens"),
    "pearson_R_pvalue": plt.get_cmap("Purples"),
    "spearman_R_pvalue": plt.get_cmap("Purples"),
    # Overall skills
    "NSE": plt.get_cmap("Spectral"),
    "KGE": plt.get_cmap("Spectral"),
}


def plot_map(da: xr.DataArray, geodata: dict, title: str, ax=None, bbox=None, 
             colorbar: bool = True, axis: str = "off", map_kwargs: dict = None, 
             cbar_kwargs: dict = None, cmap_params: dict = None):

    values = da.data.copy()
    values = np.ma.masked_invalid(values)

    # Assumes the input dimensions are lat/lon
    nlat, nlon = values.shape

    x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
        nlat, nlon, geodata=geodata
    )

    ax = get_basemap_axis(extent, ax=ax, geodata=geodata, map_kwargs=map_kwargs)

    im = ax.pcolormesh(
                x_grid,
                y_grid,
                values,
                cmap=cmap_params["cmap"],
                zorder=10,
            )

    if title:
        plt.title(title)

    # add colorbar
    if colorbar:
        if "label" not in cbar_kwargs:
            cbar_kwargs["label"] = label_from_attrs(da)
        cbar = plt.colorbar(
            im, spacing="uniform", extend="max", shrink=0.8, cax=None,
        )
        cbar.set_label(cbar_kwargs["label"])
        
        # cbar = _add_colorbar(im, ax, cbar_ax=None, cbar_kwargs=cbar_kwargs, cmap_params=cmap_params)

    if geodata is None or axis == "off":
            ax.xaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])

    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    return ax

cbar_kwargs = {"pad": 0.03, "shrink": 0.7, "label": "precip"}
cmap_params = {"cmap": plt.get_cmap("Reds")}
plot_map(ds.isel({"time":0}).precip, geodata=METADATA, title="Test", 
         cbar_kwargs=cbar_kwargs, bbox=bbox)



def plot_skill_maps(
    ds_skill: xr.Dataset,
    figs_dir: pathlib.Path,
    geodata: dict = None,
    bbox: Tuple[int] = None,
    variables: List[str] = ["precip"],
    skills: List[str]=["BIAS", "RMSE", "rSD", "pearson_R2", "error_CoV"],
    suffix: str = "",
    prefix: str = "",
):
    
    figs_dir.mkdir(exist_ok=True)

    ##------------------------------------------------------------------------.
    # Create a figure for each leadtime
    for i, leadtime in enumerate(ds_skill.leadtime.values):
        # Temporary dataset for a specific leadtime
        ds = ds_skill.sel(leadtime=leadtime)
        ##--------------------------------------------------------------------.
        # Define super title
        suptitle = "Forecast skill at lead time: {}".format(str(leadtime))

        ##--------------------------------------------------------------------.
        # Create figure
        fig, axs = plt.subplots(
            len(skills),
            len(variables),
            figsize=(15, 20),
            subplot_kw=dict(projection=geodata["projection"]),
        )
        ##--------------------------------------------------------------------.
        # Add supertitle
        fig.suptitle(suptitle, fontsize=26, y=1.05, x=0.45)
        ##--------------------------------------------------------------------.
        # Set the variable title
        for ax, var in zip(axs[0, :], variables):
            ax.set_title(var.upper(), fontsize=24, y=1.08)
        ##--------------------------------------------------------------------.
        # Display skill maps
        ax_count = 0
        axs = axs.flatten()
        for skill in skills:
            for var in variables:
                cbar_kwargs = {"pad": 0.03, "shrink": 0.7, "label": skill}
                plot_map(ds[var].sel(skill=skill), 
                         ax=axs[ax_count], 
                         geodata=geodata, 
                         title=None, 
                         cbar_kwargs=cbar_kwargs, 
                         bbox=bbox)

                axs[ax_count].outline_patch.set_linewidth(5)
                ax_count += 1
        
        ##--------------------------------------------------------------------.
        # Figure tight layout
        fig.tight_layout(pad=-2)
        plt.show()
        ##--------------------------------------------------------------------.
        # Define figure filename
        if prefix != "":
            prefix = prefix + "_"
        if suffix != "":
            suffix = "_" + suffix
        leadtime_str = str(int(leadtime / np.timedelta64(1, "h")))
        fname = prefix + "L" + leadtime_str + suffix + ".png"
        ##--------------------------------------------------------------------.
        # Save figure
        fig.savefig((figs_dir / fname), bbox_inches="tight")
        ##--------------------------------------------------------------------.
    