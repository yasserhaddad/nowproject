import pathlib
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from typing import List, Tuple, Union
from nowproject.utils.plot_precip import plot_single_precip
from pysteps.visualization.utils import proj4_to_cartopy
from pysteps.visualization.precipfields import get_colormap
from matplotlib import colors
from matplotlib.colors import Normalize

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
        ax = plot_single_precip(tmp_obs, geodata=geodata, title=title)

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