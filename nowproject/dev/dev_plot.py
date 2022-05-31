import pathlib
from typing import List, Tuple
import pandas as pd
import numpy as np
import xarray as xr

import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pysteps.visualization.precipfields import plot_precip_field, get_colormap
from pysteps.visualization.utils import proj4_to_cartopy
from nowproject.data.data_utils import prepare_data_dynamic
from nowproject.data.data_config import METADATA
from nowproject.utils.plot import (
    plot_map, 
    plot_skill_maps, 
    SKILL_CMAP_DICT, 
    plot_averaged_skills,
    plot_skills_distribution,
    plot_forecast_error_comparison
)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib import gridspec

import numpy as np
from matplotlib import cm, colors

from pysteps.visualization.utils import get_geogrid, get_basemap_axis
PRECIP_VALID_TYPES = ("intensity", "depth", "prob")
PRECIP_VALID_UNITS = ("mm/h", "mm", "dBZ")

from PIL import Image

# %load_ext autoreload
# %autoreload 2

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

# ----------------------------------------------------------------------------------------
bbox = (451000, 30000, 850000, 321000)
cbar_kwargs = {"pad": 0.03, "shrink": 0.7, "label": "precip"}
cmap_params = {"cmap": plt.get_cmap("Reds")}
plot_map(ds.isel({"time":0}).precip, geodata=METADATA, title="Test", 
         cbar_kwargs=cbar_kwargs, bbox=bbox)

crs = pyproj.CRS.from_epsg(21781)

# METADATA = {
#     "EPSG":  21781,
#     "projection": crs.to_proj4(),
#     "PROJ_parameters": crs.to_json(),
#     "x1": 451000.0,
#     "y1": 30000.0,
#     "x2": 850000.0,
#     "y2": 319000.0,
#     "xpixelsize": 1000.0,
#     "ypixelsize": 1000.0,
#     "cartesian_unit": "m",
#     "yorigin": "upper",
#     "institution": "MeteoSwiss",
#     "product": "RZC",
#     "accutime": 2.5,
#     "unit": 'mm/h',
#     "zr_a": 316.0,
#     "zr_b": 1.5
# }

# METADATA = {
#     "EPSG":  21781,
#     "projection": crs.to_proj4(),
#     "PROJ_parameters": crs.to_json(),
#     "x1": 255000.0,
#     "y1": -160000.0,
#     "x2": 894000.0,
#     "y2": 549000.0,
#     "xpixelsize": 1000.0,
#     "ypixelsize": 1000.0,
#     "cartesian_unit": "m",
#     "yorigin": "upper",
#     "institution": "MeteoSwiss",
#     "product": "RZC",
#     "accutime": 2.5,
#     "unit": 'mm/h',
#     "zr_a": 316.0,
#     "zr_b": 1.5
# }

bbox = (451000, 30000, 850000, 319000)
model_dir = pathlib.Path("/home/haddad/experiments/RNN-AR6-UNet-AvgPooling/")
ds_skill = xr.open_dataset(model_dir / "model_skills" / "deterministic_spatial_skill.nc")

skill = "BIAS"
cbar_kwargs = {"pad": 0.03, "shrink": 0.7, "label": skill}
cmap_params = {"cmap": SKILL_CMAP_DICT[skill]}
plot_map(ds_skill["feature"].isel(leadtime=10).sel(skill=skill), geodata=METADATA, 
         cbar_kwargs=cbar_kwargs, cmap_params=cmap_params, bbox=bbox)



model_dir = pathlib.Path("/home/haddad/experiments/RNN-AR6-UNet-AvgPooling/")
ds_skill = xr.open_dataset(model_dir / "model_skills" / "deterministic_spatial_skill.nc")
bbox = (470000, 60000, 835000, 300000)
# - Create spatial maps
plot_skill_maps(
    ds_skill=ds_skill,
    figs_dir=(model_dir / "figs" / "skills" / "SpatialSkill"),
    geodata=METADATA,
    bbox=bbox,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2", "error_CoV"],
    variables=["feature"],
    suffix="",
    prefix="",
)

plot_skills_distribution(ds_skill, variables=["feature"])

ds_averaged_skill = xr.open_dataset(model_dir / "model_skills" / "deterministic_global_skill.nc")

plot_averaged_skills(ds_averaged_skill, variables=["feature"])


# ----------------------------------------------------------------------------------------
crs = pyproj.CRS.from_epsg(21781)
geodata = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 485500.0,
    "y1": 75500.0,
    "x2": 830500.0,
    "y2": 300500.0,
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

model_dir = pathlib.Path("/home/haddad/experiments/RNN-AR6-EPDNet-AvgPooling-Patches-LogNormalizeScaler-MSE-10epochs/")
forecast_zarr_fpath = (
        model_dir / "model_predictions" / "forecast_chunked" / "test_forecasts.zarr"
    )

ds_forecasts = xr.open_zarr(forecast_zarr_fpath)

data_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/")
boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
ds_obs = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                    boundaries=boundaries)

ds_forecasts_small = ds_forecasts.isel(forecast_reference_time=10).copy()

plot_forecast_error_comparison(model_dir / "figs" / "gifs", ds_forecasts_small, ds_obs, geodata=geodata)

model_dir_2 = pathlib.Path("/home/haddad/experiments/RNN-AR6-UNet-MaxPooling-Patches-LogNormalizeScaler-MSE-15epochs/")
forecast_zarr_fpath = (
        model_dir_2 / "model_predictions" / "forecast_chunked" / "test_forecasts.zarr"
    )

ds_forecasts_2 = xr.open_zarr(forecast_zarr_fpath)

ds_forecasts_2_small = ds_forecasts_2.isel(forecast_reference_time=10).copy()

# ----------------------------------------------------------------------------------------
# Adri's request
def plot_precip(
    precip,
    ptype="intensity",
    ax=None,
    geodata=None,
    units="mm/h",
    bbox=None,
    colorscale="pysteps",
    probthr=None,
    title=None,
    colorbar=True,
    axis="on",
    cax=None,
    map_kwargs=None,
):
    """
    Function to plot a precipitation intensity or probability field with a
    colorbar.
    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes
    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html
    Parameters
    ----------
    precip: array-like
        Two-dimensional array containing the input precipitation field or an
        exceedance probability map.
    ptype: {'intensity', 'depth', 'prob'}, optional
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'depth' = precipitation depth (accumulation) field,
        'prob' = exceedance probability field.
    geodata: dictionary or None, optional
        Optional dictionary containing geographical information about
        the field. Required is map is not None.
        If geodata is not None, it must contain the following key-value pairs:
        .. tabularcolumns:: |p{1.5cm}|L|
        +-----------------+---------------------------------------------------+
        |        Key      |                  Value                            |
        +=================+===================================================+
        |    projection   | PROJ.4-compatible projection definition           |
        +-----------------+---------------------------------------------------+
        |    x1           | x-coordinate of the lower-left corner of the data |
        |                 | raster                                            |
        +-----------------+---------------------------------------------------+
        |    y1           | y-coordinate of the lower-left corner of the data |
        |                 | raster                                            |
        +-----------------+---------------------------------------------------+
        |    x2           | x-coordinate of the upper-right corner of the     |
        |                 | data raster                                       |
        +-----------------+---------------------------------------------------+
        |    y2           | y-coordinate of the upper-right corner of the     |
        |                 | data raster                                       |
        +-----------------+---------------------------------------------------+
        |    yorigin      | a string specifying the location of the first     |
        |                 | element in the data raster w.r.t. y-axis:         |
        |                 | 'upper' = upper border, 'lower' = lower border    |
        +-----------------+---------------------------------------------------+
    units : {'mm/h', 'mm', 'dBZ'}, optional
        Units of the input array. If ptype is 'prob', this specifies the unit of
        the intensity threshold.
    bbox : tuple, optional
        Four-element tuple specifying the coordinates of the bounding box. Use
        this for plotting a subdomain inside the input grid. The coordinates are
        of the form (lower left x, lower left y ,upper right x, upper right y).
        If 'geodata' is not None, the bbox is in map coordinates, otherwise
        it represents image pixels.
    colorscale : {'pysteps', 'STEPS-BE', 'BOM-RF3'}, optional
        Which colorscale to use. Applicable if units is 'mm/h', 'mm' or 'dBZ'.
    probthr : float, optional
        Intensity threshold to show in the color bar of the exceedance
        probability map.
        Required if ptype is "prob" and colorbar is True.
    title : str, optional
        If not None, print the title on top of the plot.
    colorbar : bool, optional
        If set to True, add a colorbar on the right side of the plot.
    axis : {'off','on'}, optional
        Whether to turn off or on the x and y axis.
    cax : Axes_ object, optional
        Axes into which the colorbar will be drawn. If no axes is provided
        the colorbar axes are created next to the plot.
    Other parameters
    ----------------
    map_kwargs: dict
        Optional parameters that need to be passed to
        :py:func:`pysteps.visualization.basemaps.plot_geography`.
    Returns
    -------
    ax : fig Axes_
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    """

    if map_kwargs is None:
        map_kwargs = {}

    if ptype not in PRECIP_VALID_TYPES:
        raise ValueError(
            f"Invalid precipitation type '{ptype}'."
            f"Supported: {str(PRECIP_VALID_TYPES)}"
        )

    if units not in PRECIP_VALID_UNITS:
        raise ValueError(
            f"Invalid precipitation units '{units}."
            f"Supported: {str(PRECIP_VALID_UNITS)}"
        )

    if ptype == "prob" and colorbar and probthr is None:
        raise ValueError("ptype='prob' but probthr not specified")

    if len(precip.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    # Assumes the input dimensions are lat/lon
    nlat, nlon = precip.shape

    x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
        nlat, nlon, geodata=geodata
    )

    # ax = get_basemap_axis(extent, ax=ax, geodata=geodata, map_kwargs=map_kwargs)
    ax = plot_map_cartopy(proj4_to_cartopy(geodata["projection"]), 
                          extent=extent,
                          cartopy_scale="50m")

    precip = np.ma.masked_invalid(precip)
    # plot rainfield
    if regular_grid:
        im = _plot_field(precip, ax, ptype, units, colorscale, extent, origin=origin)
    else:
        im = _plot_field(
            precip, ax, ptype, units, colorscale, extent, x_grid=x_grid, y_grid=y_grid
        )

    plt.title(title)

    # add colorbar
    if colorbar:
        # get colormap and color levels
        _, _, clevs, clevs_str = get_colormap(ptype, units, colorscale)
        if ptype in ["intensity", "depth"]:
            extend = "max"
        else:
            extend = "neither"
        cbar = plt.colorbar(
            im, ticks=clevs, spacing="uniform", extend=extend, shrink=0.8, cax=cax
        )
        if clevs_str is not None:
            cbar.ax.set_yticklabels(clevs_str)

        if ptype == "intensity":
            cbar.set_label(f"Precipitation intensity ({units})")
        elif ptype == "depth":
            cbar.set_label(f"Precipitation depth [{units}]")
        else:
            cbar.set_label(f"P(R > {probthr:.1f} {units})")

    if geodata is None or axis == "off":
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])

    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    return ax


def plot_map_cartopy(
    crs,
    extent,
    cartopy_scale,
    figsize=(8, 5),
    drawlonlatlines=False,
    drawlonlatlabels=True,
    lw=0.5,
    subplot=None,
):
    """
    Plot coastlines, countries, rivers and meridians/parallels using cartopy.
    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html
    Parameters
    ----------
    crs: object
        Instance of a crs class defined in cartopy.crs.
        It can be created using utils.proj4_to_cartopy.
    extent: scalars (left, right, bottom, top)
        The coordinates of the bounding box.
    drawlonlatlines: bool
        Whether to plot longitudes and latitudes.
    drawlonlatlabels: bool, optional
        If set to True, draw longitude and latitude labels. Valid only if
        'drawlonlatlines' is True.
    cartopy_scale: {'10m', '50m', '110m'}
        The scale (resolution) of the map. The available options are '10m',
        '50m', and '110m'.
    lw: float
        Line width.
    subplot: tuple of int (nrows, ncols, index) or SubplotSpec_ instance, optional
        The subplot where to place the basemap.
        By the default, the basemap will replace the current axis.
    Returns
    -------
    ax: axes
        Cartopy axes. Compatible with matplotlib.
    """

    if subplot is None:
        ax = plt.gca(figsize=figsize)
    else:
        if isinstance(subplot, gridspec.SubplotSpec):
            ax = plt.subplot(subplot, projection=crs)
        else:
            ax = plt.subplot(*subplot, projection=crs)

    if not isinstance(ax, GeoAxesSubplot):
        ax = plt.subplot(ax.get_subplotspec(), projection=crs)
        # cax.clear()
        ax.set_axis_off()

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "ocean",
            scale="50m" if cartopy_scale == "10m" else cartopy_scale,
            edgecolor="none",
            facecolor=np.array([0.59375, 0.71484375, 0.8828125]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            scale=cartopy_scale,
            edgecolor="none",
            facecolor=np.array([0.9375, 0.9375, 0.859375]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "coastline",
            scale=cartopy_scale,
            edgecolor="black",
            facecolor="none",
            linewidth=lw,
        ),
        zorder=15,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "lakes",
            scale=cartopy_scale,
            edgecolor="none",
            facecolor=np.array([0.59375, 0.71484375, 0.8828125]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "rivers_lake_centerlines",
            scale=cartopy_scale,
            edgecolor=np.array([0.59375, 0.71484375, 0.8828125]),
            facecolor="none",
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural",
            "admin_0_boundary_lines_land",
            scale=cartopy_scale,
            edgecolor="black",
            facecolor="none",
            linewidth=lw,
        ),
        zorder=15,
    )
    if cartopy_scale in ["10m", "50m"]:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "reefs",
                scale="10m",
                edgecolor="black",
                facecolor="none",
                linewidth=lw,
            ),
            zorder=15,
        )
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "minor_islands",
                scale="10m",
                edgecolor="black",
                facecolor="none",
                linewidth=lw,
            ),
            zorder=15,
        )

    if drawlonlatlines:
        grid_lines = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=drawlonlatlabels, dms=True
        )
        grid_lines.top_labels = grid_lines.right_labels = False
        grid_lines.y_inline = grid_lines.x_inline = False
        grid_lines.rotate_labels = False

    ax.set_extent(extent, crs)

    return ax

def _plot_field(
    precip, ax, ptype, units, colorscale, extent, origin=None, x_grid=None, y_grid=None
):
    precip = precip.copy()

    # Get colormap and color levels
    cmap, norm, _, _ = get_colormap(ptype, units, colorscale)

    if (x_grid is None) or (y_grid is None):
        im = ax.imshow(
            precip,
            cmap=cmap,
            norm=norm,
            extent=extent,
            interpolation="nearest",
            origin=origin,
            zorder=10,
        )
    else:
        im = ax.pcolormesh(
            x_grid,
            y_grid,
            precip,
            cmap=cmap,
            norm=norm,
            zorder=10,
        )

    return im

def plot_obs(figs_dir: pathlib.Path,
             ds_obs: xr.Dataset,
             geodata: dict = None,
             bbox: Tuple[int] = None,
             save_gif: bool = True,
             fps: int = 4,
            ):
    
    figs_dir.mkdir(exist_ok=True)
    (figs_dir / "tmp").mkdir(exist_ok=True)
    # Load in memory
    ds_obs = ds_obs.load()
    var = list(ds_obs.data_vars.keys())[0]
    # Retrieve common variables to plot 
    cmap, norm, clevs, clevs_str = get_colormap("intensity")
    cmap_params = {"cmap": cmap, "norm": norm}
    cbar_params = { "ticks": clevs,
                    "shrink": 0.7, 
                    "extend": "both",
                    "label": "Precipitation intensity (mm/h)",
                    "clevs_str": clevs_str}
    pil_frames = []

    for i, time in enumerate(ds_obs.time.values):
        time_str = str(time.astype('datetime64[s]'))
        filepath = figs_dir / "tmp" / f"{time_str}.png"
        ##--------------------------------------------------------------------.
        # Create figure
        fig, ax = plt.subplots(
            figsize=(8, 5),
            # subplot_kw={'projection': proj4_to_cartopy(METADATA["projection"])}
        )
        ##---------------------------------------------------------------------.
        # Plot each variable
        tmp_obs = ds_obs[var].isel(time=i)
        
        # Plot obs 
        # ax = plot_map(tmp_obs,
        #                 ax=ax, 
        #                 geodata=geodata, 
        #                 title=None, 
        #                 colorbar=True,
        #                 cmap_params=cmap_params,
        #                 cbar_params=cbar_params,
        #                 bbox=bbox)
        ax = plot_precip_field(tmp_obs.values, geodata=geodata, ax=ax)
        ax.set_title("RZC, Time: {}".format(time_str))
        # ax.outline_patch.set_linewidth(1)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if save_gif:
            pil_frames.append(Image.open(filepath).convert("P",palette=Image.ADAPTIVE))
            # fig.canvas.draw()
            # pil_frames.append(Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()).convert("P"))

    if save_gif:
        date = str(pd.to_datetime(ds_obs.time.values[0]).date())
        pil_frames[0].save(
            figs_dir / f"{date}.gif",
            format="gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=1 / fps * 1000,  # ms
            loop=False,
        )

crs = pyproj.CRS.from_epsg(21781)
geodata = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 485000.0,
    "y1": 75000.0,
    "x2": 830000.0,
    "y2": 300000.0,
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

data_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/")
boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
ds_obs = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                    boundaries=boundaries)

figs_dir = pathlib.Path("/home/haddad/adri_figs/")

time_start = np.datetime64('2021-06-22T16:00:00.000000000')
time_end = np.datetime64('2021-06-22T20:00:00.000000000')

ds_sel = ds_obs.sel(time=slice(time_start, time_end))

plot_obs(figs_dir, ds_sel, geodata=geodata, fps=6)

METADATA = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 255000.0,
    "y1": -160000.0,
    "x2": 965000.0,
    "y2": 480000.0,
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

figs_full_dir = pathlib.Path("/home/haddad/adri_figs/full/")

data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr")
ds_full_sel = data_dynamic.sel(time=slice(time_start, time_end))
plot_obs(figs_full_dir, ds_sel, geodata=METADATA, fps=6)

# ----------------------------------------------------------------------------------------
# Debug figs
crs = pyproj.CRS.from_epsg(21781)
geodata = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 485000.0,
    "y1": 75000.0,
    "x2": 830000.0,
    "y2": 300000.0,
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

data_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/")
boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
ds_obs = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                    boundaries=boundaries)

figs_dir = pathlib.Path("/home/haddad/debug_figs/")

time_start = np.datetime64('2021-06-28T13:30:00.000000000')
time_end = np.datetime64('2021-06-28T17:30:00.000000000')

ds_sel = ds_obs.sel(time=slice(time_start, time_end))

plot_obs(figs_dir, ds_sel, geodata=geodata, fps=6)

METADATA = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 255000.0,
    "y1": -160000.0,
    "x2": 965000.0,
    "y2": 480000.0,
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

figs_full_dir = pathlib.Path("/home/haddad/debug_figs/full/")

data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr")
ds_full_sel = data_dynamic.sel(time=slice(time_start, time_end))
plot_obs(figs_full_dir, ds_sel, geodata=METADATA, fps=6)


# ----
from nowproject.utils.plot_map import plot_obs
from nowproject.data.data_config import BOTTOM_LEFT_COORDINATES

METADATA = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 255000.0,
    "y1": -160000.0,
    "x2": 965000.0,
    "y2": 480000.0,
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

time_start = np.datetime64('2021-06-22T16:00:00.000000000')
time_end = np.datetime64('2021-06-22T20:00:00.000000000')

data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr")

da = data_dynamic.sel(time=slice(time_start, time_end))[["feature"]]

figs_full_dir = pathlib.Path("/home/haddad/debug_figs/full/")

plot_obs(figs_full_dir, da, geodata=METADATA, fps=6)

geodata = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 486000.0,
    "y1": 76000.0,
    "x2": 831000.0,
    "y2": 301000.0,
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

boundaries = {
    "x": slice(485, 831),
    "y": slice(301, 75),
}
data_dynamic_ch = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                       boundaries=boundaries)

data_dynamic_ch = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr")
data_dynamic_ch = data_dynamic_ch.sel(x=slice(485, 831))
data_dynamic_ch = data_dynamic_ch.sel(y=np.arange(301, 75, -1))


def xr_sel_coords_between(data, **kwargs):
    for k, slc in kwargs.items():
        if not isinstance(slc, slice):
             raise TypeError("Expects slice objects")
        # TODO check coord is 1 dim and exist 
        # 
        if slc.stop >= slc.start:
            data = data.sel({k: slc})
        else:
            isel_idx = np.where(np.logical_and(data[k].data > slc.stop, data[k].data <= slc.start))[0]
            data = data.isel({k: isel_idx})
    return data 

data_dynamic_ch = xr_sel_coords_between(data_dynamic_ch, **boundaries)

data_dynamic_ch.y.data 



da_ch = data_dynamic_ch.sel(time=slice(time_start, time_end))[["feature"]]
# assert np.allclose(da_ch.sel(x=569.5, y=211.5).feature.data, da.sel(x=569.5, y=211.5).feature.data)
# assert xr.testing.assert_allclose(da_ch.sel(x=569.5, y=211.5).feature.load(), da.sel(x=569.5, y=211.5).feature.load())
# assert xr.testing.assert_identical(da_ch.sel(x=569.5, y=211.5).feature.load(), da.sel(x=569.5, y=211.5).feature.load())
# assert xr.testing.assert_equal(da_ch.sel(x=569.5, y=211.5).feature.load(), da.sel(x=569.5, y=211.5).feature.load())

# da_ch.sel(x=569.5, y=211.5).feature.load()
# da.sel(x=569.5, y=211.5).feature.load()

figs_dir = pathlib.Path("/home/haddad/debug_figs/")

plot_obs(figs_dir, da_ch, geodata=geodata, fps=6)

# --------

from nowproject.data.data_config import METADATA_CH
from nowproject.utils.plot_map import plot_obs

figs_dir = pathlib.Path("/home/haddad/debug_figs/")

boundaries = {
    "x": slice(485, 831),
    "y": slice(301, 75),
}
data_dynamic_ch = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                       boundaries=boundaries)

time_start = np.datetime64('2021-06-22T16:00:00.000000000')
time_end = np.datetime64('2021-06-22T20:00:00.000000000')

da = data_dynamic_ch.sel(time=slice(time_start, time_end))[["feature"]]

plot_obs(figs_dir, da, METADATA_CH, fps=6)