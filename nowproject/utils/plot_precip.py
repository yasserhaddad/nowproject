import pyproj
import numpy as np
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
    extent,
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

    # ax.set_extent(extent, crs)

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

    ax = _plot_map_cartopy(proj4_to_cartopy(geodata["projection"]), 
                           extent=extent,
                           cartopy_scale="50m",
                           ax=ax)

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
            cbar.set_label(f"Precipitation depth ({units})")
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