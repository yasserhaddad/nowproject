from re import M
from typing import Tuple, Union
import matplotlib
import pyproj
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, Colormap
from matplotlib.image import AxesImage

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxesSubplot

from pysteps.visualization.utils import proj4_to_cartopy
from pysteps.visualization.precipfields import get_colormap
from pysteps.visualization.utils import get_geogrid

PRECIP_VALID_TYPES = ("intensity", "depth", "prob")
PRECIP_VALID_UNITS = ("mm/h", "mm", "dBZ")


def _plot_map_cartopy(
    crs: ccrs.Projection,
    figsize: tuple = (8, 5),
    cartopy_scale: str = "50m",
    ax: Axes = None,
    drawlonlatlines: bool = False,
    drawlonlatlabels: bool = True,
    lw: float = 0.5
):
    """Plot coastlines, countries, rivers and meridians/parallels using cartopy.

    Parameters
    ----------
    crs : ccrs.Projection
        Instance of a crs class defined in cartopy.crs.
        It can be created using utils.proj4_to_cartopy.
    figsize : tuple, optional
        Figure size if ax is not specified, by default (8, 5)
    cartopy_scale : str, optional
        The scale (resolution) of the map. The available options are '10m',
        '50m', and '110m', by default "50m"
    ax : Axes, optional
        Axes object to plot the map on, by default None
    drawlonlatlines : bool, optional
        If true, plot longitudes and latitudes, by default False
    drawlonlatlabels : bool, optional
        If true, draw longitude and latitude labels. Valid only if
        'drawlonlatlines' is true, by default True
    lw : float, optional
        Line width, by default 0.5

    Returns
    -------
    Axes
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


def plot_single_precip(da: xr.DataArray, geodata: dict, ax: Axes = None, ptype="intensity", 
                       units="mm/h", colorscale="pysteps", figsize=(8,5), title: str = None, 
                       colorbar: bool = True, drawlonlatlines: bool = False, 
                       extent: Tuple[Union[int,float]] = None, probthr: float = None,
                       norm: Normalize = None, 
                       cmap: Union[Colormap, str] = None) -> Tuple[GeoAxesSubplot, AxesImage]:
    """Plot a single precipitation event (one timestep).

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the data to plot, with two
        spatial dimensions (y-x, lat-lon)
    geodata : dict
        Metadata containing at least the projection, the 
        corners of the plot ('x1', 'y1', 'x2', 'y2') and 
        'yorigin'
    ax : Axes, optional
        Axes object to plot the map on, by default None
    ptype : str, optional
        Type of the map to plot. Options : {'intensity', 
        'depth', 'prob'}, by default "intensity"
    units : str, optional
        Units of the input array. Options : {'mm/h', 
        'mm', 'dBZ'}, by default "mm/h"
    colorscale : str, optional
        Colorscale to use. Options : {'pysteps', 'STEPS-BE', 
        'BOM-RF3'}, by default "pysteps"
    figsize : tuple, optional
        Figure size if ax is not specified, by default (8,5)
    title : str, optional
        If not None, print the title on top of the plot, 
        by default None
    colorbar : bool, optional
        If true, add colorbar on the right side of the plot, 
        by default True
    drawlonlatlines : bool, optional
        If true, plot longitudes and latitudes, by default False
    extent : Tuple[Union[int,float]], optional
        bounding box in data coordinates that the image will fill, 
        by default None
    probthr : float, optional
        Intensity threshold to show in the color bar of the 
        exceedance probability map. Required if ptype is “prob” 
        and colorbar is True, by default None
    norm : Normalize, optional
        Normalize instance used to scale scalar data to the [0, 1] 
        range before mapping to colors using cmap, by default None
    cmap : Union[Colormap, str], optional
        Colormap instance or registered colormap name used to map 
        scalar data to colors, by default None

    Returns
    -------
    Tuple[GeoAxesSubplot, AxesImage] 
        The subplot 
    """
    if not cmap and not norm:
        cmap, norm, clevs, clevs_str = get_colormap(ptype, units, colorscale)
    else:
        clevs, clevs_str = None, None
    crs_ref = proj4_to_cartopy(geodata["projection"])
    crs_proj = crs_ref
    
    if ax is None:
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
        cbar_kwargs=cbar_kwargs if colorbar else None,
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

    return ax, p


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