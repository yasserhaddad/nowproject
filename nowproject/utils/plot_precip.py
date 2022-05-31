from re import M
from typing import Tuple, Union
import pyproj
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxesSubplot

from pysteps.visualization.utils import proj4_to_cartopy
from pysteps.visualization.precipfields import get_colormap
from pysteps.visualization.utils import get_geogrid

PRECIP_VALID_TYPES = ("intensity", "depth", "prob")
PRECIP_VALID_UNITS = ("mm/h", "mm", "dBZ")


def _plot_map_cartopy(
    crs,
    figsize=(8, 5),
    cartopy_scale="50m",
    ax=None,
    drawlonlatlines=False,
    drawlonlatlabels=True,
    lw=0.5
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
    if not ax:
        ax = plt.gca(figsize=figsize)

    if not isinstance(ax, GeoAxesSubplot):
        ax = plt.subplot(ax.get_subplotspec(), projection=crs)
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

    return ax


def plot_single_precip(da: xr.DataArray, geodata: dict, ptype="intensity", units="mm/h", 
                       colorscale="pysteps", figsize=(8,5), title: str = None, 
                       colorbar: bool = True, drawlonlatlines: bool = False, 
                       extent: Tuple[Union[int,float]] = None, probthr: float = None):
    cmap, norm, clevs, clevs_str = get_colormap(ptype, units, colorscale)
    crs_ref = proj4_to_cartopy(geodata["projection"])
    crs_proj = crs_ref

    _, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={'projection': crs_proj}
        )

    cbar_kwargs = {
        "ticks": clevs,
        "spacing": "uniform",
        "extend": "max",
        "shrink": 0.8,
    }

    p = da.plot.imshow(
        ax=ax,
        transform=crs_ref,
        cmap=cmap,
        norm=norm, 
        interpolation="nearest",
        add_colorbar=colorbar,
        cbar_kwargs=cbar_kwargs,
        zorder=1,
    )

    if title:
        ax.set_title(title)
    
    if colorbar:
        if clevs_str is not None:
            p.colorbar.ax.set_yticklabels(clevs_str)
        if ptype == "intensity":
            p.colorbar.set_label(f"Precipitation intensity ({units})")
        elif ptype == "depth":
            p.colorbar.set_label(f"Precipitation depth ({units})")
        else:
            p.colorbar.set_label(f"P(R > {probthr:.1f} {units})")

    p.axes = _plot_map_cartopy(crs_proj, 
                               cartopy_scale="50m",
                               drawlonlatlines=drawlonlatlines,
                               ax=p.axes)
    
    if extent:
        p.axes.set_extent(extent, crs_ref)

    return ax


def plot_multifaceted_grid_precip(da: xr.DataArray, geodata: dict, col: str, col_wrap: int,
                                  ptype="intensity", units="mm/h", 
                                  colorscale="pysteps", figsize=(12,8), title: str = None, 
                                  colorbar: bool = True, drawlonlatlines: bool = False, 
                                  extent: Tuple[Union[int,float]] = None, probthr: float = None):
    
    cmap, norm, clevs, clevs_str = get_colormap(ptype, units, colorscale)
    crs_ref = proj4_to_cartopy(geodata["projection"])
    crs_proj = crs_ref

    if ptype == "intensity":
        cbar_label = f"Precipitation intensity ({units})"
    elif ptype == "depth":
        cbar_label = f"Precipitation depth ({units})"
    else:
        cbar_label = f"P(R > {probthr:.1f} {units})"

    cbar_kwargs = {
        "ticks": clevs,
        "spacing": "uniform",
        "extend": "max",
        "shrink": 0.8,
        "label": cbar_label
    }

    p = da.plot.imshow(
        subplot_kws={'projection': crs_proj},
        figsize=figsize,
        transform=crs_ref,
        col=col, col_wrap=col_wrap,
        cmap=cmap,
        norm=norm, 
        interpolation="nearest",
        add_colorbar=colorbar,
        cbar_kwargs=cbar_kwargs,
        zorder=1,
    )

    if title:
        p.fig.suptitle(title, x=0.4, y=1.0, fontsize="x-large")

    for ax in p.axes.flatten():
        ax = _plot_map_cartopy(crs_proj, 
                               cartopy_scale="50m",
                               drawlonlatlines = drawlonlatlines,
                               ax=ax)
    
        if extent:
            ax.set_extent(extent, crs_ref)

    return p