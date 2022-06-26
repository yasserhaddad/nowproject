import calendar
import pathlib

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

from PIL import Image
from typing import List, Tuple, Union
from nowproject.utils.plot_precip import plot_single_precip
from pysteps.visualization.utils import proj4_to_cartopy
from pysteps.visualization.precipfields import get_colormap
from matplotlib import colors
from matplotlib.colors import Normalize

from skimage.morphology import square
import matplotlib.patches as mpatches
from nowproject.data.patches_utils import (
    get_areas_labels,
    get_patch_per_label,
    patch_stats_fun,
    xr_get_areas_labels,
    get_slice_size,
)

class FixPointNormalize(matplotlib.colors.Normalize):
    """ 
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint 
    somewhere in the middle of the colormap.
    This may be useful for a `terrain` map, to set the "sea level" 
    to a color in the blue/turquise range. 
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def rescale_spatial_axes(ds: Union[xr.Dataset, xr.DataArray],
                         spatial_dims: List[str] = ["y", "x"],
                         scale_factor: int = 1000) -> Union[xr.Dataset, xr.DataArray]:
    
    ds = ds.assign_coords({spatial_dims[0]: ds[spatial_dims[0]].data*scale_factor})
    ds = ds.assign_coords({spatial_dims[1]: ds[spatial_dims[1]].data*scale_factor})

    return ds


def get_colormap_error():
    clevs = [-60, - 30, -16, -8, -4, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 4, 8, 16, 30, 60]
    clevs_str = [str(clev) for clev in clevs]
    norm = colors.BoundaryNorm(boundaries=clevs, ncolors=len(clevs)+1)
    # cmap = plt.get_cmap("RdBu_r").reversed()
    cmap = plt.get_cmap("Spectral")
    return cmap, norm, clevs, clevs_str


def plot_patches(figs_dir: pathlib.Path,
                 da_obs: xr.DataArray):

    intensity = da_obs.data.copy() 
    min_intensity_threshold = 0.1
    max_intensity_threshold = 300
    min_area_threshold = 36
    max_area_threshold = np.Inf
    
    footprint_buffer = square(10)
    
    patch_size = (128,128)
    centered_on = "centroid"
    mask_value = 0

    da_labels, n_labels, counts = xr_get_areas_labels(da_obs,  
                                                        min_intensity_threshold=min_intensity_threshold, 
                                                        max_intensity_threshold=max_intensity_threshold, 
                                                        min_area_threshold=min_area_threshold, 
                                                        max_area_threshold=max_area_threshold, 
                                                        footprint_buffer=footprint_buffer)

    labels = da_labels.data
    list_patch_slices, patch_statistics = get_patch_per_label(labels, 
                                                            intensity, 
                                                            patch_size = patch_size, 
                                                            # centered_on = "max",
                                                            # centered_on = "centroid",
                                                            centered_on = "center_of_mass",
                                                            patch_stats_fun = patch_stats_fun, 
                                                            mask_value = mask_value, 
                                                            )

    cmap, norm, clevs, clevs_str = get_colormap("intensity", "mm/h", "pysteps")

    ref_time = str(da_obs.time.values.astype('datetime64[s]'))
    cbar_kwargs = {
        "ticks": clevs,
        "spacing": "uniform",
        "extend": "max",
        "shrink": 0.9
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    p = da_obs.plot.imshow(ax=ax, norm=norm, cmap=cmap, add_colorbar=True, cbar_kwargs=cbar_kwargs)
    p.colorbar.ax.set_yticklabels(clevs_str)
    p.colorbar.set_label(f"Precipitation intensity (mm/h)")
    ax.set_title(f"Observation at time : {ref_time}")
    ax.set_axis_off()
    plt.savefig(figs_dir / "observation.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap_qualitative = plt.get_cmap("Spectral")
    cmap_qualitative.set_under(color="white")
    im = plt.imshow(labels, vmin=1, cmap=cmap_qualitative, interpolation="none")

    values = np.unique(labels.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    legends = ["No precipitation" if value == 0 else f"Event {value}" for value in values]
    patches = [mpatches.Patch(color=colors[i], label=legends[i]) for i in range(len(values)) ]
    plt.legend(handles=patches, loc=4)
    plt.grid(None)
    ax.set_axis_off()
    plt.title(f"Precipitation events detected at time : {ref_time}")
    plt.savefig(figs_dir / "precipitation_events.png", dpi=300, bbox_inches='tight')

    list_patch_upper_left_idx = [[slc.start for slc in list_slc] for list_slc in list_patch_slices]
    list_patch_upper_left_idx

    cbar_kwargs = {
        "ticks": clevs,
        "spacing": "uniform",
        "extend": "max",
        "shrink": 0.8
    }
    # Plot all bounding boxes 
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    p = plt.imshow(intensity, norm=norm, cmap=cmap)
    cb = plt.colorbar(ax=ax, **cbar_kwargs)
    cb.ax.set_yticklabels(clevs_str)
    cb.set_label(f"Precipitation intensity (mm/h)")
    for y, x in list_patch_upper_left_idx:
        rect = plt.Rectangle((x, y), patch_size[0], patch_size[0], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f"Patches extracted at time : {ref_time}")
    ax.set_axis_off()
    plt.savefig(figs_dir / "patches.png", dpi=300, bbox_inches='tight')



def plot_obs(figs_dir: pathlib.Path,
             ds_obs: xr.Dataset,
             geodata: dict = None,
             bbox: Tuple[int] = None,
             save_gif: bool = True,
             fps: int = 4,
             figsize=(8, 5)
            ):
    
    figs_dir.mkdir(exist_ok=True)
    (figs_dir / "tmp").mkdir(exist_ok=True)
    # Load in memory
    ds_obs = rescale_spatial_axes(ds_obs.load(), scale_factor=1000)

    var = list(ds_obs.data_vars.keys())[0]
    pil_frames = []

    for i, time in enumerate(ds_obs.time.values):
        time_str = str(time.astype('datetime64[s]'))
        filepath = figs_dir / "tmp" / f"{time_str}.png"
        ##---------------------------------------------------------------------.
        # Plot each variable
        tmp_obs = ds_obs[var].isel(time=i)
        
        # Plot obs 
        title = "RZC, Time: {}".format(time_str)
        ax, p = plot_single_precip(tmp_obs, geodata=geodata, title=title, figsize=figsize)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if save_gif:
            pil_frames.append(Image.open(filepath).convert("P",palette=Image.ADAPTIVE))
 

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


def plot_stats_map(figs_dir: pathlib.Path, 
                   ds_stats: xr.Dataset, 
                   geodata: dict = None,
                   figsize=(8, 5),
                   time_coord="month",
                   save_gif: bool = True,
                   fps: int = 4,
                   title_prefix=None,
                   ):
    figs_dir.mkdir(exist_ok=True)
    (figs_dir / "tmp").mkdir(exist_ok=True)

    ds_stats = rescale_spatial_axes(ds_stats.load(), scale_factor=1000)
    vars = list(ds_stats.data_vars.keys())
    pil_frames = []

    for var in vars:
        for i, time in enumerate(ds_stats[time_coord].values):
            filepath = figs_dir / "tmp" / f"{var}_{time}.png"
            ##---------------------------------------------------------------------.
            # Plot each variable
            tmp_obs = ds_stats[var].isel({time_coord: i})
            
            # Plot obs 
            title = f"{title_prefix}, {var.capitalize()}" if title_prefix else f"{var.capitalize()}"
            if time_coord == "month":
                title += f", {calendar.month_name[time]}"
            else:
                title += f", {time}"
            ax = plot_single_precip(tmp_obs, geodata=geodata, title=title, figsize=figsize)

            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            if save_gif:
                pil_frames.append(Image.open(filepath).convert("P",palette=Image.ADAPTIVE))
    

        if save_gif:
            pil_frames[0].save(
                figs_dir / f"{var}.gif",
                format="gif",
                save_all=True,
                append_images=pil_frames[1:],
                duration=1 / fps * 1000,  # ms
                loop=False,
            )


def plot_forecast_error_comparison(figs_dir: pathlib.Path,
                                   ds_forecast: xr.Dataset,
                                   ds_obs: xr.Dataset,
                                   geodata: dict = None,
                                   aspect_cbar: int = 40,
                                   save_gif: bool = True,
                                   fps: int = 4,
                                   suptitle_prefix: str = "",
                                 ):
    
    figs_dir.mkdir(exist_ok=True)

    forecast_reference_time = str(ds_forecast['forecast_reference_time'].values.astype('datetime64[s]'))
    (figs_dir / f"tmp_{forecast_reference_time}").mkdir(exist_ok=True)
    # Load data into memory
    ds_forecast = rescale_spatial_axes(ds_forecast.load(), scale_factor=1000)

    # Retrieve valid time 
    valid_time = ds_forecast['forecast_reference_time'].values + ds_forecast['leadtime'].values
    ds_forecast = ds_forecast.assign_coords({'time': ('leadtime', valid_time)})
    ds_forecast = ds_forecast.swap_dims({'leadtime': 'time'})

    # Subset observations and load in memory
    ds_obs = ds_obs.sel(time=ds_forecast['time'].values)
    ds_obs = rescale_spatial_axes(ds_obs.load(), scale_factor=1000)

    # Compute error 
    ds_error = ds_forecast - ds_obs 

    # Create a dictionary with relevant infos  
    ds_dict = {"pred": ds_forecast, "obs": ds_obs, "error": ds_error}

    _, norm, clevs, clevs_str = get_colormap("intensity", "mm/h", "pysteps")
    cmap_error, norm_error, clevs_error, clevs_str_error = get_colormap_error()

    # Retrieve common variables to plot 
    variables = list(ds_forecast.data_vars.keys())
    pil_frames = []
    for i, leadtime in enumerate(ds_forecast.leadtime.values):
        filepath = figs_dir / f"tmp_{forecast_reference_time}" / f"L{i:02d}.png"
        ##--------------------------------------------------------------------.
        # Define super title
        suptitle = "{}Forecast reference time: {}, Lead time: {}".format(
            suptitle_prefix,
            forecast_reference_time, 
            str(leadtime.astype("timedelta64[m]"))
        )
        # Create figure
        fig, axs = plt.subplots(
            len(variables),
            3,
            figsize=(18, 4*len(variables)),
            subplot_kw={'projection': proj4_to_cartopy(geodata["projection"])}
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
            _ = plot_single_precip(tmp_obs,
                                      ax=axs[ax_count], 
                                      geodata=geodata, 
                                      title=None, 
                                      colorbar=False)
            axs[ax_count].set_title(None)
            axs[ax_count].outline_patch.set_linewidth(1)

            tmp_pred = ds_dict['pred'][var].isel(time=i)
            # Plot 
            _, p_2 = plot_single_precip(tmp_pred,
                                      ax=axs[ax_count+1], 
                                      geodata=geodata, 
                                      title=None, 
                                      colorbar=False)
            axs[ax_count+1].set_title(None)
            axs[ax_count+1].outline_patch.set_linewidth(1)
            # - Add state colorbar
            cbar = fig.colorbar(p_2, ax=axs[[ax_count, ax_count+1]], 
                                ticks=clevs,
                                spacing="uniform",
                                orientation="horizontal", 
                                extend = 'both',
                                aspect=aspect_cbar)      
            cbar.ax.set_xticklabels(clevs_str) 
            cbar.set_label("Precipitation intensity (mm/h)")
            cbar.ax.xaxis.set_label_position('top')

            tmp_error = ds_dict['error'][var].isel(time=i)
            # norm = Normalize(vmin=clevs_error[0], vmax=clevs_error[-1])
            _, p_3 = plot_single_precip(tmp_error,
                                      ax=axs[ax_count+2], 
                                      geodata=geodata, 
                                      title=None, 
                                      colorbar=False,
                                      norm=norm_error,
                                      cmap=cmap_error)
            axs[ax_count+2].set_title(None)
            axs[ax_count+2].outline_patch.set_linewidth(1)
            # - Add error colorbar
            # cb = plt.colorbar(e_p, ax=axs[ax_count+2], orientation="horizontal") # pad=0.15)
            # cb.set_label(label=var.upper() + " Error") # size='large', weight='bold'
            cbar_err = fig.colorbar(p_3, ax=axs[ax_count+2],
                                    ticks=clevs_error,
                                    orientation="horizontal",
                                    extend = 'both',
                                    aspect = aspect_cbar/2)  
            cbar_err.ax.set_xticklabels(clevs_str_error)    
            cbar_err.set_label("Precipitation intensity error")
            cbar_err.ax.xaxis.set_label_position('top')
            # Add plot labels 
            # if ax_count == 0: 
            axs[ax_count].set_title("Observed")     
            axs[ax_count+1].set_title("Predicted")  
            axs[ax_count+2].set_title("Error")
            # Update ax_count 
            ax_count += 3

        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        if save_gif:
            pil_frames.append(Image.open(filepath).convert("P",palette=Image.ADAPTIVE))
        

    if save_gif:
        pil_frames[0].save(
            figs_dir / f"{forecast_reference_time}.gif",
            format="gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=1 / fps * 1000,  # ms
            loop=False,
        )


def plot_forecast_comparison(figs_dir: pathlib.Path,
                             ds_forecast: xr.Dataset,
                             ds_obs: xr.Dataset,
                             geodata: dict = None,
                             aspect_cbar: int = 40,
                             save_gif: bool = True,
                             fps: int = 4,
                             suptitle_prefix: str = "",
                            ):
    
    figs_dir.mkdir(exist_ok=True)

    forecast_reference_time = str(ds_forecast['forecast_reference_time'].values.astype('datetime64[s]'))
    (figs_dir / f"tmp_{forecast_reference_time}").mkdir(exist_ok=True)
    # Load data into memory
    ds_forecast = rescale_spatial_axes(ds_forecast.load(), scale_factor=1000)

    # Retrieve valid time 
    valid_time = ds_forecast['forecast_reference_time'].values + ds_forecast['leadtime'].values
    ds_forecast = ds_forecast.assign_coords({'time': ('leadtime', valid_time)})
    ds_forecast = ds_forecast.swap_dims({'leadtime': 'time'})


    # Subset observations and load in memory
    ds_obs = ds_obs.sel(time=ds_forecast['time'].values)
    ds_obs = rescale_spatial_axes(ds_obs.load(), scale_factor=1000)

    # Create a dictionary with relevant infos  
    ds_dict = {"pred": ds_forecast, "obs": ds_obs}

    _, _, clevs, clevs_str = get_colormap("intensity", "mm/h", "pysteps")

    # Retrieve common variables to plot 
    variables = list(ds_forecast.data_vars.keys())
    pil_frames = []
    for i, leadtime in enumerate(ds_forecast.leadtime.values):
        filepath = figs_dir / f"tmp_{forecast_reference_time}" / f"L{i:02d}.png"
        ##--------------------------------------------------------------------.
        # Define super title
        suptitle = "{}Forecast reference time: {}, Lead time: {}".format(
            suptitle_prefix,
            forecast_reference_time, 
            str(leadtime.astype("timedelta64[m]"))
        )
        # Create figure
        fig, axs = plt.subplots(
            len(variables),
            2,
            figsize=(14, 4*len(variables)),
            subplot_kw={'projection': proj4_to_cartopy(geodata["projection"])}
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
            _ = plot_single_precip(tmp_obs,
                                      ax=axs[ax_count], 
                                      geodata=geodata, 
                                      title=None, 
                                      colorbar=False)
            axs[ax_count].set_title(None)
            axs[ax_count].outline_patch.set_linewidth(1)

            tmp_pred = ds_dict['pred'][var].isel(time=i)
            # Plot 
            _, p_2 = plot_single_precip(tmp_pred,
                                      ax=axs[ax_count+1], 
                                      geodata=geodata, 
                                      title=None, 
                                      colorbar=False)
            axs[ax_count+1].set_title(None)
            axs[ax_count+1].outline_patch.set_linewidth(1)
            # - Add state colorbar
            cbar = fig.colorbar(p_2, ax=axs[[ax_count, ax_count+1]], 
                                ticks=clevs,
                                spacing="uniform",
                                orientation="horizontal", 
                                extend = 'both',
                                aspect=aspect_cbar)       
            cbar.set_label("Precipitation intensity (mm/h)")
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.set_xticklabels(clevs_str)

            # Add plot labels 
            # if ax_count == 0: 
            axs[ax_count].set_title("Observed")     
            axs[ax_count+1].set_title("Predicted")  
            # Update ax_count 
            ax_count += 2

        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        if save_gif:
            pil_frames.append(Image.open(filepath).convert("P",palette=Image.ADAPTIVE))
        

    if save_gif:
        pil_frames[0].save(
            figs_dir / f"{forecast_reference_time}.gif",
            format="gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=1 / fps * 1000,  # ms
            loop=False,
        )