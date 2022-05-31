import pathlib
from typing import List, Tuple
import numpy as np
import xarray as xr

import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
import matplotlib.pyplot as plt
from pysteps.visualization.precipfields import plot_precip_field, get_colormap
from nowproject.data.data_config import METADATA
from cartopy.mpl.geoaxes import GeoAxesSubplot
from PIL import Image

from xarray.plot.utils import _add_colorbar, label_from_attrs
from pysteps.visualization.utils import get_geogrid, get_basemap_axis, proj4_to_cartopy

# import seaborn as sns
# sns.set_theme()

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

SKILL_CBAR_EXTEND_DICT = {
    "error_CoV": "both",
    "obs_CoV": "both",
    "pred_CoV": "both",
    # Magnitude
    "BIAS": "both",
    "relBIAS": "both",
    "percBIAS": "both",
    "MAE": "max",
    "relMAE": "max",
    "percMAE": "max",
    "MSE": "max",
    "relMSE": "max",
    "RMSE": "max",
    "relRMSE": "max",
    # Average
    "rMean": "both",
    "diffMean": "both",
    # Variability
    "rSD": "both",
    "diffSD": "both",
    "rCoV": "max",
    "diffCoV": "both",
    # Correlation
    "pearson_R": "neither",
    "pearson_R2": "neither",
    "spearman_R": "neither",
    "spearman_R2": "neither",
    "pearson_R2_pvalue": "neither",
    "spearman_R2_pvalue": "max",
    # Overall skills
    "NSE": "min",
    "KGE": "neither",
}

SKILL_YLIM_DICT = {
    "BIAS": (-4, 4),
    "RMSE": (0, 8),
    "rSD": (0.6, 1.4),
    "pearson_R2": (0, 1),
    "KGE": (-1.1, 1.1),
    "NSE": (-0.1, 1.1),
    "relBIAS": (-0.01, 0.01),
    "percBIAS": (-2.5, 2.5),
    "percMAE": (0, 2.5),
    "error_CoV": (-40, 40),
    "MAE": (0.5, 2.5),
    "diffSD": (-1.1, 1.1),
    "SSIM": (-1.1, 1.1),
    "POD": (-0.1, 1.1),
    "FAR": (-0.1, 1.1),
    "FA": (-0.1, 1.1),
    "ACC": (-0.1, 1.1),
    "CSI": (-0.1, 1.1),
    "FB": (-4, 4),
    "HSS": (-1.1, 1.1),
    "HK": (-0.1, 1.1),
    "GSS": (-0.1, 1.1),
    "SEDI": (-0.1, 1.1),
    "MCC": (-0.1, 1.1),
    "F1": (-0.1, 1.1)
}

def _plot_map_cartopy(
    crs,
    extent,
    figsize=(8, 5),
    cartopy_scale="50m",
    ax=None,
    drawlonlatlines=False,
    drawlonlatlabels=True,
    lw=1
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



def _plot_field(values: np.ndarray, ax: Axes, cmap: Colormap, norm: Normalize = None, x_grid: np.ndarray = None, 
                y_grid: np.ndarray = None, extent: Tuple[float] = None, origin: str = None,
                vmin: float = None, vmax: float = None):
     
    if (x_grid is None) or (y_grid is None):
        im = ax.imshow(
            values,
            cmap=cmap,
            norm=norm,
            extent=extent,
            interpolation="nearest",
            origin=origin,
            zorder=10,
            vmin=vmin, 
            vmax=vmax
        )
    else:
        im = ax.pcolormesh(
                x_grid,
                y_grid,
                values,
                cmap=cmap,
                norm=norm,
                zorder=10,
                vmin=vmin,
                vmax=vmax
            )
    
    return im

def plot_map(da: xr.DataArray, geodata: dict, title: str = None, ax=None, bbox=None, 
             vmin: float = None, vmax: float = None, colorbar: bool = True, 
             axis: str = "off", map_kwargs: dict = None, cbar_params: dict = None, 
             cmap_params: dict = None, return_im: bool = False):

    values = da.data.copy()
    values = np.ma.masked_invalid(values)

    # Assumes the input dimensions are lat/lon
    nlat, nlon = values.shape

    x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
        nlat, nlon, geodata=geodata
    )

    # ax = get_basemap_axis(extent, ax=ax, geodata=geodata, map_kwargs=map_kwargs)
    ax = _plot_map_cartopy(proj4_to_cartopy(geodata["projection"]), extent, ax=ax)
    
    if not cmap_params:
        cmap, norm, _, _ = get_colormap("intensity")
    else:
        cmap = cmap_params.get("cmap", plt.get_cmap("Reds"))
        norm = None if not cmap_params else cmap_params.get("norm", None)
    
    if regular_grid:
        im = _plot_field(values, ax, cmap, norm=norm, extent=extent, origin=origin, vmin=vmin, vmax=vmax)
    else:
        im = _plot_field(values, ax, cmap, norm=norm, extent=extent, x_grid=x_grid, y_grid=y_grid, 
                         vmin=vmin, vmax=vmax)

    if title:
        plt.title(title)

    # add colorbar
    if colorbar:
        if not cbar_params:
            _, _, clevs, clevs_str = get_colormap("intensity")
            cbar_params = {
                "extend": "max",
                "shrink": 0.8,
                "label": "",
                "ticks": clevs,
                "clevs_str": clevs_str
            }
        if "label" not in cbar_params:
            cbar_params["label"] = label_from_attrs(da)
        ticks = None if not cbar_params else cbar_params.get("ticks", None)
        clevs_str = cbar_params.get("clevs_str", None)
        cbar = plt.colorbar(
            im, 
            ax=ax, 
            ticks=ticks,
            spacing="uniform",
            extend=cbar_params.get("extend", "max"), 
            shrink=cbar_params.get("shrink", 0.8), 
            cax=None,
        )
        cbar.set_label(cbar_params["label"])
        if clevs_str is not None:
            cbar.ax.set_yticklabels(clevs_str)
        
        # cbar = _add_colorbar(im, ax, cbar_ax=None, cbar_kwargs=cbar_kwargs, cmap_params=cmap_params)

    if geodata is None or axis == "off":
            ax.xaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])

    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    if return_im:
        return im
    else:
        return ax


# def plot_precip(precip, )


def plot_forecast_error_comparison(figs_dir: pathlib.Path,
                                   ds_forecast: xr.Dataset,
                                   ds_obs: xr.Dataset,
                                   geodata: dict = None,
                                   bbox: Tuple[int] = None,
                                   aspect_cbar: int = 40,
                                   save_gif: bool = True,
                                   fps: int = 4,
                                 ):
    
    figs_dir.mkdir(exist_ok=True)

    forecast_reference_time = str(ds_forecast['forecast_reference_time'].values.astype('datetime64[s]'))
    (figs_dir / f"tmp_{forecast_reference_time}").mkdir(exist_ok=True)
    # Load data into memory
    ds_forecast = ds_forecast.load()

    # Retrieve valid time 
    valid_time = ds_forecast['forecast_reference_time'].values + ds_forecast['leadtime'].values
    ds_forecast = ds_forecast.assign_coords({'time': ('leadtime', valid_time)})
    ds_forecast = ds_forecast.swap_dims({'leadtime': 'time'})

    # Subset observations and load in memory
    ds_obs = ds_obs.sel(time=ds_forecast['time'].values)
    ds_obs = ds_obs.load()

    # Compute error 
    ds_error = ds_forecast - ds_obs 

    # Create a dictionary with relevant infos  
    ds_dict = {"pred": ds_forecast, "obs": ds_obs, "error": ds_error}

    # Retrieve common variables to plot 
    variables = list(ds_forecast.data_vars.keys())
    cmap, norm, clevs, clevs_str = get_colormap("intensity")
    cmap_params = {"cmap": cmap, "norm": norm, "ticks": clevs}
    pil_frames = []
    for i, leadtime in enumerate(ds_forecast.leadtime.values):
        filepath = figs_dir / f"tmp_{forecast_reference_time}" / f"L{i:02d}.png"
        ##--------------------------------------------------------------------.
        # Define super title
        suptitle = "Forecast reference time: {}, Lead time: {}".format(
            forecast_reference_time, 
            str(leadtime.astype("timedelta64[m]"))
        )
        # Create figure
        fig, axs = plt.subplots(
            len(variables),
            3,
            figsize=(18, 4*len(variables)),
            subplot_kw={'projection': proj4_to_cartopy(METADATA["projection"])}
        )

        fig.suptitle(suptitle, y=1.05)
        ##---------------------------------------------------------------------.
        # Initialize
        axs = axs.flatten()
        ax_count = 0
        ##---------------------------------------------------------------------.
        # Plot each variable
        for var in variables:
            tmp_obs = ds_dict['obs'][var].isel(time=i)
            # Plot obs 
            im_1 = plot_map(tmp_obs,
                          ax=axs[ax_count], 
                          geodata=geodata, 
                          title=None, 
                          colorbar=False,
                          cmap_params=cmap_params, 
                          bbox=bbox,
                          vmin=0,
                          vmax=30,
                          return_im=True)
            axs[ax_count].set_title(None)
            axs[ax_count].outline_patch.set_linewidth(1)

            tmp_pred = ds_dict['pred'][var].isel(time=i)
            # Plot 
            im_2 = plot_map(tmp_pred,
                          ax=axs[ax_count+1], 
                          geodata=geodata, 
                          title=None, 
                          colorbar=False,
                          cmap_params=cmap_params, 
                          bbox=bbox,
                          vmin=0,
                          vmax=30,
                          return_im=True)
            axs[ax_count+1].set_title(None)
            axs[ax_count+1].outline_patch.set_linewidth(1)
            # - Add state colorbar
            cbar = fig.colorbar(im_2, ax=axs[[ax_count, ax_count+1]], 
                                orientation="horizontal", 
                                extend = 'both',
                                aspect=aspect_cbar)       
            cbar.set_label(var.upper())
            cbar.ax.xaxis.set_label_position('top')

            error_cmap_params = {"cmap": "Spectral"}
            tmp_error = ds_dict['error'][var].isel(time=i)
            im_3 = plot_map(tmp_error,
                          ax=axs[ax_count+2], 
                          geodata=geodata, 
                          title=None, 
                          colorbar=False,
                          cmap_params=error_cmap_params, 
                          bbox=bbox,
                          vmin=-20,
                          vmax=20,
                          return_im=True)
            axs[ax_count+2].set_title(None)
            axs[ax_count+2].outline_patch.set_linewidth(1)
            # - Add error colorbar
            # cb = plt.colorbar(e_p, ax=axs[ax_count+2], orientation="horizontal") # pad=0.15)
            # cb.set_label(label=var.upper() + " Error") # size='large', weight='bold'
            cbar_err = fig.colorbar(im_3, ax=axs[ax_count+2],
                                    orientation="horizontal",
                                    extend = 'both',
                                    aspect = aspect_cbar/2)      
            cbar_err.set_label(var.upper() + " Error")
            cbar_err.ax.xaxis.set_label_position('top')
            # Add plot labels 
            # if ax_count == 0: 
            axs[ax_count].set_title("Observed")     
            axs[ax_count+1].set_title("Predicted")  
            axs[ax_count+2].set_title("Error")
            # Update ax_count 
            ax_count += 3

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if save_gif:
            pil_frames.append(Image.open(filepath).convert("P",palette=Image.ADAPTIVE))
            # fig.canvas.draw()
            # pil_frames.append(Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()).convert("P"))

    if save_gif:
        pil_frames[0].save(
            figs_dir / f"{forecast_reference_time}.gif",
            format="gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=1 / fps * 1000,  # ms
            loop=False,
        )


def plot_comparison_maps(ds_forecast_list: List[xr.Dataset],
                         forecast_labels: List[str],
                         ds_obs: xr.Dataset,
                         figs_dir: pathlib.Path,
                         geodata: dict = None,
                         bbox: Tuple[int] = None,
                         aspect_cbar: int = 40,
                         save_gif: bool = True,
                         fps: int = 4,
                        ):
    figs_dir.mkdir(exist_ok=True)

    forecast_reference_time = str(ds_forecast_list[0]['forecast_reference_time'].values.astype('datetime64[s]'))
    (figs_dir / f"tmp_{forecast_reference_time}").mkdir(exist_ok=True)
    # Load data into memory
    ds_forecast_list = [ds_forecast.sel(leadtime=ds_forecast_list[0]['leadtime'].values)\
                                   .load() for ds_forecast in ds_forecast_list]

    # Retrieve valid time 
    valid_time = ds_forecast_list[0]['forecast_reference_time'].values + ds_forecast_list[0]['leadtime'].values
    ds_forecast_list = [ds_forecast.assign_coords({'time': ('leadtime', valid_time)}).swap_dims({'leadtime': 'time'})\
                        for ds_forecast in ds_forecast_list]

    # Subset observations and load in memory
    ds_obs = ds_obs.sel(time=ds_forecast_list[0]['time'].values)
    ds_obs = ds_obs.load()

    # Retrieve common variables to plot 
    variables = list(set.intersection(*map(set, [ds_forecast.data_vars.keys() for ds_forecast in ds_forecast_list])))
    
    cmap, norm, clevs, clevs_str = get_colormap("intensity")
    cmap_params = {"cmap": cmap, "norm": norm, "ticks": clevs}
    pil_frames = []
    for i, leadtime in enumerate(ds_forecast_list[0].leadtime.values):
        filepath = figs_dir / f"tmp_{forecast_reference_time}" / f"L{i:02d}.png"
        ##--------------------------------------------------------------------.
        # Define super title
        suptitle = "Forecast reference time: {}, Lead time: {}".format(
            forecast_reference_time, 
            str(leadtime.astype("timedelta64[m]"))
        )
        # Create figure
        fig, axs = plt.subplots(
            len(variables),
            len(ds_forecast_list) + 1,
            figsize=(6*(len(ds_forecast_list) + 1), 4*len(variables)),
            subplot_kw={'projection': proj4_to_cartopy(METADATA["projection"])}
        )

        fig.suptitle(suptitle, y=1.05)
        ##---------------------------------------------------------------------.
        # Initialize
        axs = axs.flatten()
        ax_count = 0
        ##---------------------------------------------------------------------.
        # Plot each variable
        for var in variables:
            # Plot obs 
            tmp_obs = ds_obs[var].isel(time=i)
            im_1 = plot_map(tmp_obs,
                          ax=axs[ax_count], 
                          geodata=geodata, 
                          title=None, 
                          colorbar=False,
                          cmap_params=cmap_params, 
                          bbox=bbox,
                          vmin=0,
                          vmax=30,
                          return_im=True)
            axs[ax_count].set_title(None)
            axs[ax_count].outline_patch.set_linewidth(1)

            for j, ds_forecast in enumerate(ds_forecast_list):
                tmp_pred = ds_forecast[var].isel(time=i)
                # Plot 
                im_forecast = plot_map(tmp_pred,
                                ax=axs[ax_count+j+1], 
                                geodata=geodata, 
                                title=None, 
                                colorbar=False,
                                cmap_params=cmap_params, 
                                bbox=bbox,
                                vmin=0,
                                vmax=30,
                                return_im=True)
                axs[ax_count+j+1].set_title(None)
                axs[ax_count+j+1].outline_patch.set_linewidth(1)
            # - Add state colorbar
            cbar = fig.colorbar(im_forecast, ax=axs[[ax_count + j for j in range(len(ds_forecast_list) +1)]], 
                                orientation="horizontal", 
                                extend = 'both',
                                aspect=aspect_cbar)       
            cbar.set_label(var.upper())
            cbar.ax.xaxis.set_label_position('top')

            # Add plot labels 
            # if ax_count == 0: 
            axs[ax_count].set_title("Observed")   
            for j, title in enumerate(forecast_labels): 
                axs[ax_count+j+1].set_title(title) 
            # Update ax_count 
            ax_count += len(ds_forecast_list) + 1

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if save_gif:
            pil_frames.append(Image.open(filepath).convert("P", palette=Image.ADAPTIVE))

    if save_gif:
        pil_frames[0].save(
            figs_dir / f"{forecast_reference_time}.gif",
            format="gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=1 / fps * 1000,  # ms
            loop=False,
        )


def plot_skill_maps(
    ds_skill: xr.Dataset,
    figs_dir: pathlib.Path,
    geodata: dict = None,
    bbox: Tuple[int] = None,
    variables: List[str] = ["feature"],
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
        suptitle = "Forecast skill at lead time: {}".format(str(leadtime.astype("timedelta64[m]")))

        ##--------------------------------------------------------------------.
        # Create figure
        fig, axs = plt.subplots(
            len(skills),
            len(variables),
            figsize=(15, 20),
            subplot_kw={'projection': proj4_to_cartopy(METADATA["projection"])}
        )
        ##--------------------------------------------------------------------.
        # Add supertitle
        fig.suptitle(suptitle, fontsize=26, y=1.05, x=0.6)
        ##--------------------------------------------------------------------.
        # Set the variable title
        for ax, var in zip(axs, variables):
            ax.set_title(var.upper(), fontsize=24, y=1.08)
        ##--------------------------------------------------------------------.
        # Display skill maps
        ax_count = 0
        axs = axs.flatten()
        for skill in skills:
            for var in variables:
                cbar_params = { 
                    "shrink": 0.7, 
                    "extend": SKILL_CBAR_EXTEND_DICT[skill],
                    "label": skill
                }
                cmap_params = {"cmap": SKILL_CMAP_DICT[skill]}
                ax = plot_map(ds[var].sel(skill=skill), 
                              ax=axs[ax_count], 
                              geodata=geodata, 
                              title=None, 
                              cbar_params=cbar_params,
                              cmap_params=cmap_params, 
                              bbox=bbox,
                              vmin=SKILL_YLIM_DICT.get(skill, (None, None))[0],
                              vmax=SKILL_YLIM_DICT.get(skill, (None, None))[1])

                axs[ax_count].outline_patch.set_linewidth(2)
                ax_count += 1
        
        ##--------------------------------------------------------------------.
        # Figure tight layout
        # plt.subplots_adjust(
        #     bottom=0.1,  
        #     top=0.9, 
        #     wspace=0.4, 
        #     hspace=0.4
        # )
        fig.tight_layout()
        # plt.show()
        ##--------------------------------------------------------------------.
        # Define figure filename
        if prefix != "":
            prefix = prefix + "_"
        if suffix != "":
            suffix = "_" + suffix
        leadtime_str = "{:02d}".format((int(leadtime / np.timedelta64(1, "m"))))
        fname = prefix + "L" + leadtime_str + suffix + ".png"
        ##--------------------------------------------------------------------.
        # Save figure
        fig.savefig((figs_dir / fname), bbox_inches="tight")
        ##--------------------------------------------------------------------.


def plot_averaged_skill(
    ds_averaged_skill, skill="RMSE", variables=["precip"], n_leadtimes=None
):
    if not n_leadtimes:
        n_leadtimes = len(ds_averaged_skill.leadtime)
    # Plot first n_leadtimes
    ds_averaged_skill = ds_averaged_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_averaged_skill["leadtime"].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[m]")]
    # Create figure
    fig, axs = plt.subplots(1, len(variables), figsize=(15, 4))
    axs = [axs] if len(variables) < 2 else axs.flatten()    

    for ax, var in zip(axs, variables):
        # Plot global average skill
        ax.plot(leadtimes, ds_averaged_skill[var].sel(skill=skill).values)
        ##------------------------------------------------------------------.
        # Add best skill line
        if skill in [
            "relBIAS",
            "BIAS",
            "percBIAS",
            "diffMean",
            "diffSD",
            "diffCoV",
            "error_CoV",
        ]:
            ax.axhline(y=0, linestyle="solid", color="gray", alpha=0.2)
        elif skill in ["rSD", "rMean", "rCoV"]:
            ax.axhline(y=1, linestyle="solid", color="gray", alpha=0.2)
        ##------------------------------------------------------------------.
        # Add labels
        ax.set_ylim(SKILL_YLIM_DICT.get(skill, (None, None)))
        ax.set_xlabel("Leadtime (min)")
        ax.set_ylabel(skill)
        # Set axis appearance
        ax.margins(x=0, y=0)
        # Set xticks
        ax.set_xticks(leadtimes[::2])
        ax.set_xticklabels(leadtimes[::2])
        ##------------------------------------------------------------------.
        # Add title
        ax.set_title(var.upper())
        ##------------------------------------------------------------------.
    fig.tight_layout()
    return fig


def plot_averaged_skills(
    ds_averaged_skill,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2", "KGE", "error_CoV"],
    variables=["precip"],
    n_leadtimes=None,
):
    if not n_leadtimes:
        n_leadtimes = len(ds_averaged_skill.leadtime)
    # Plot first n_leadtimes
    ds_averaged_skill = ds_averaged_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_averaged_skill["leadtime"].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[m]")]
    # Create figure
    fig, axs = plt.subplots(len(skills), len(variables), figsize=(17, 18))
    # Initialize axes
    ax_i = 0
    axs = axs.flatten()
    for skill in skills:
        for var in variables:
            # Plot global average skill
            axs[ax_i].plot(leadtimes, ds_averaged_skill[var].sel(skill=skill).values)
            ##------------------------------------------------------------------.
            # Add best skill line
            if skill in [
                "relBIAS",
                "BIAS",
                "percBIAS",
                "diffMean",
                "diffSD",
                "diffCoV",
                "error_CoV",
            ]:
                axs[ax_i].axhline(y=0, linestyle="solid", color="gray", alpha=0.2)
            elif skill in ["rSD", "rMean", "rCoV"]:
                axs[ax_i].axhline(y=1, linestyle="solid", color="gray", alpha=0.2)
            ##------------------------------------------------------------------.
            # Add labels
            axs[ax_i].set_ylim(SKILL_YLIM_DICT.get(skill, (None, None)))
            axs[ax_i].set_xlabel("Leadtime (min)")
            axs[ax_i].set_ylabel(skill)
            # Set axis appearance
            axs[ax_i].margins(x=0, y=0)
            # Set xticks
            axs[ax_i].set_xticks(leadtimes[::2])
            axs[ax_i].set_xticklabels(leadtimes[::2])
            ##------------------------------------------------------------------.
            # Add title
            if ax_i < len(variables):
                axs[ax_i].set_title(var.upper())
            ##------------------------------------------------------------------.
            # Update ax count
            ax_i += 1
    # Figure tight layout
    fig.tight_layout()
    return fig


def plot_skills_distribution(
    ds_skill,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2", "KGE", "error_CoV"],
    variables=["precip"],
    n_leadtimes=None,
):
    if not n_leadtimes:
        n_leadtimes = len(ds_skill.leadtime)
    # Plot first n_leadtimes
    ds_skill = ds_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_skill["leadtime"].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[m]")]
    # Create figure
    fig, axs = plt.subplots(len(skills), len(variables), figsize=(17, 18))
    # Initialize axes
    ax_i = 0
    axs = axs.flatten()
    for skill in skills:
        for var in variables:
            # Plot skill distribution
            tmp_boxes = [
                ds_skill[var].sel(skill=skill).values[i, :].reshape(-1)
                for i in range(len(ds_skill[var].sel(skill=skill).values))
            ]
            tmp_boxes = [d[~np.isnan(d)] for d in tmp_boxes]
            axs[ax_i].boxplot(tmp_boxes, showfliers=False)
            ##------------------------------------------------------------------.
            # Add best skill line
            if skill in [
                "relBIAS",
                "BIAS",
                "percBIAS",
                "diffMean",
                "diffSD",
                "diffCoV",
                "error_CoV",
            ]:
                axs[ax_i].axhline(y=0, linestyle="solid", color="gray")
            elif skill in ["rSD", "rMean", "rCoV"]:
                axs[ax_i].axhline(y=1, linestyle="solid", color="gray")
            ##------------------------------------------------------------------.
            # Add labels
            # axs[ax_i].set_ylim(SKILL_YLIM_DICT.get(skill, (None, None)))
            axs[ax_i].set_xlabel("Leadtime (min)")
            axs[ax_i].set_ylabel(skill)
            axs[ax_i].set_xticklabels(leadtimes)
            axs[ax_i].tick_params(axis="both", which="major", labelsize=14)
            ##------------------------------------------------------------------.
            # Violin plots
            # import seaborn as sns
            # da = ds_skill[var].sel(skill=skill)
            # da.to_dataframe().reset_index()
            # ax = sns.boxplot(x=df.time.dt.hour, y=name, data=df)
            ##------------------------------------------------------------------.
            # Add title
            if ax_i < len(variables):
                axs[ax_i].set_title(var.upper())
            ##------------------------------------------------------------------.
            # Update ax count
            ax_i += 1
    # Figure tight layout
    fig.tight_layout()
    return fig