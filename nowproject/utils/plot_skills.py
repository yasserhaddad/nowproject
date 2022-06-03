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
    "RMSE": (0, 4),
    "rSD": (0.6, 1.4),
    "pearson_R2": (0, 1),
    "KGE": (-1.05, 1.05),
    "NSE": (-0.05, 1.05),
    "relBIAS": (-0.01, 0.01),
    "percBIAS": (-2.5, 2.5),
    "percMAE": (0, 2.5),
    "error_CoV": (-40, 40),
    "MAE": (0.5, 2.5),
    "diffSD": (-1.05, 1.05),
    "SSIM": (-1.05, 1.05),
    "POD": (-0.05, 1.05),
    "FAR": (-0.05, 1.05),
    "FA": (-0.05, 1.05),
    "ACC": (-0.05, 1.05),
    "CSI": (-0.05, 1.05),
    "FB": (-4, 4),
    "HSS": (-1.05, 1.05),
    "HK": (-0.05, 1.05),
    "GSS": (-0.05, 1.05),
    "SEDI": (-0.05, 1.05),
    "MCC": (-0.05, 1.05),
    "F1": (-0.05, 1.05),
    "FSS": (-0.05, 1.05)
}


# def plot_comparison_maps(ds_forecast_list: List[xr.Dataset],
#                          forecast_labels: List[str],
#                          ds_obs: xr.Dataset,
#                          figs_dir: pathlib.Path,
#                          geodata: dict = None,
#                          bbox: Tuple[int] = None,
#                          aspect_cbar: int = 40,
#                          save_gif: bool = True,
#                          fps: int = 4,
#                         ):
#     figs_dir.mkdir(exist_ok=True)

#     forecast_reference_time = str(ds_forecast_list[0]['forecast_reference_time'].values.astype('datetime64[s]'))
#     (figs_dir / f"tmp_{forecast_reference_time}").mkdir(exist_ok=True)
#     # Load data into memory
#     ds_forecast_list = [ds_forecast.sel(leadtime=ds_forecast_list[0]['leadtime'].values)\
#                                    .load() for ds_forecast in ds_forecast_list]

#     # Retrieve valid time 
#     valid_time = ds_forecast_list[0]['forecast_reference_time'].values + ds_forecast_list[0]['leadtime'].values
#     ds_forecast_list = [ds_forecast.assign_coords({'time': ('leadtime', valid_time)}).swap_dims({'leadtime': 'time'})\
#                         for ds_forecast in ds_forecast_list]

#     # Subset observations and load in memory
#     ds_obs = ds_obs.sel(time=ds_forecast_list[0]['time'].values)
#     ds_obs = ds_obs.load()

#     # Retrieve common variables to plot 
#     variables = list(set.intersection(*map(set, [ds_forecast.data_vars.keys() for ds_forecast in ds_forecast_list])))
    
#     cmap, norm, clevs, clevs_str = get_colormap("intensity")
#     cmap_params = {"cmap": cmap, "norm": norm, "ticks": clevs}
#     pil_frames = []
#     for i, leadtime in enumerate(ds_forecast_list[0].leadtime.values):
#         filepath = figs_dir / f"tmp_{forecast_reference_time}" / f"L{i:02d}.png"
#         ##--------------------------------------------------------------------.
#         # Define super title
#         suptitle = "Forecast reference time: {}, Lead time: {}".format(
#             forecast_reference_time, 
#             str(leadtime.astype("timedelta64[m]"))
#         )
#         # Create figure
#         fig, axs = plt.subplots(
#             len(variables),
#             len(ds_forecast_list) + 1,
#             figsize=(6*(len(ds_forecast_list) + 1), 4*len(variables)),
#             subplot_kw={'projection': proj4_to_cartopy(METADATA["projection"])}
#         )

#         fig.suptitle(suptitle, y=1.05)
#         ##---------------------------------------------------------------------.
#         # Initialize
#         axs = axs.flatten()
#         ax_count = 0
#         ##---------------------------------------------------------------------.
#         # Plot each variable
#         for var in variables:
#             # Plot obs 
#             tmp_obs = ds_obs[var].isel(time=i)
#             im_1 = plot_map(tmp_obs,
#                           ax=axs[ax_count], 
#                           geodata=geodata, 
#                           title=None, 
#                           colorbar=False,
#                           cmap_params=cmap_params, 
#                           bbox=bbox,
#                           vmin=0,
#                           vmax=30,
#                           return_im=True)
#             axs[ax_count].set_title(None)
#             axs[ax_count].outline_patch.set_linewidth(1)

#             for j, ds_forecast in enumerate(ds_forecast_list):
#                 tmp_pred = ds_forecast[var].isel(time=i)
#                 # Plot 
#                 im_forecast = plot_map(tmp_pred,
#                                 ax=axs[ax_count+j+1], 
#                                 geodata=geodata, 
#                                 title=None, 
#                                 colorbar=False,
#                                 cmap_params=cmap_params, 
#                                 bbox=bbox,
#                                 vmin=0,
#                                 vmax=30,
#                                 return_im=True)
#                 axs[ax_count+j+1].set_title(None)
#                 axs[ax_count+j+1].outline_patch.set_linewidth(1)
#             # - Add state colorbar
#             cbar = fig.colorbar(im_forecast, ax=axs[[ax_count + j for j in range(len(ds_forecast_list) +1)]], 
#                                 orientation="horizontal", 
#                                 extend = 'both',
#                                 aspect=aspect_cbar)       
#             cbar.set_label(var.upper())
#             cbar.ax.xaxis.set_label_position('top')

#             # Add plot labels 
#             # if ax_count == 0: 
#             axs[ax_count].set_title("Observed")   
#             for j, title in enumerate(forecast_labels): 
#                 axs[ax_count+j+1].set_title(title) 
#             # Update ax_count 
#             ax_count += len(ds_forecast_list) + 1

#         plt.savefig(filepath, dpi=300, bbox_inches='tight')
#         if save_gif:
#             pil_frames.append(Image.open(filepath).convert("P", palette=Image.ADAPTIVE))

#     if save_gif:
#         pil_frames[0].save(
#             figs_dir / f"{forecast_reference_time}.gif",
#             format="gif",
#             save_all=True,
#             append_images=pil_frames[1:],
#             duration=1 / fps * 1000,  # ms
#             loop=False,
#         )


# def plot_skill_maps(
#     ds_skill: xr.Dataset,
#     figs_dir: pathlib.Path,
#     geodata: dict = None,
#     bbox: Tuple[int] = None,
#     variables: List[str] = ["feature"],
#     skills: List[str]=["BIAS", "RMSE", "rSD", "pearson_R2", "error_CoV"],
#     suffix: str = "",
#     prefix: str = "",
# ):
    
#     figs_dir.mkdir(exist_ok=True)

#     ##------------------------------------------------------------------------.
#     # Create a figure for each leadtime
#     for i, leadtime in enumerate(ds_skill.leadtime.values):
#         # Temporary dataset for a specific leadtime
#         ds = ds_skill.sel(leadtime=leadtime)
#         ##--------------------------------------------------------------------.
#         # Define super title
#         suptitle = "Forecast skill at lead time: {}".format(str(leadtime.astype("timedelta64[m]")))

#         ##--------------------------------------------------------------------.
#         # Create figure
#         fig, axs = plt.subplots(
#             len(skills),
#             len(variables),
#             figsize=(15, 20),
#             subplot_kw={'projection': proj4_to_cartopy(METADATA["projection"])}
#         )
#         ##--------------------------------------------------------------------.
#         # Add supertitle
#         fig.suptitle(suptitle, fontsize=26, y=1.05, x=0.6)
#         ##--------------------------------------------------------------------.
#         # Set the variable title
#         for ax, var in zip(axs, variables):
#             ax.set_title(var.upper(), fontsize=24, y=1.08)
#         ##--------------------------------------------------------------------.
#         # Display skill maps
#         ax_count = 0
#         axs = axs.flatten()
#         for skill in skills:
#             for var in variables:
#                 cbar_params = { 
#                     "shrink": 0.7, 
#                     "extend": SKILL_CBAR_EXTEND_DICT[skill],
#                     "label": skill
#                 }
#                 cmap_params = {"cmap": SKILL_CMAP_DICT[skill]}
#                 ax = plot_map(ds[var].sel(skill=skill), 
#                               ax=axs[ax_count], 
#                               geodata=geodata, 
#                               title=None, 
#                               cbar_params=cbar_params,
#                               cmap_params=cmap_params, 
#                               bbox=bbox,
#                               vmin=SKILL_YLIM_DICT.get(skill, (None, None))[0],
#                               vmax=SKILL_YLIM_DICT.get(skill, (None, None))[1])

#                 axs[ax_count].outline_patch.set_linewidth(2)
#                 ax_count += 1
        
#         ##--------------------------------------------------------------------.
#         # Figure tight layout
#         # plt.subplots_adjust(
#         #     bottom=0.1,  
#         #     top=0.9, 
#         #     wspace=0.4, 
#         #     hspace=0.4
#         # )
#         fig.tight_layout()
#         # plt.show()
#         ##--------------------------------------------------------------------.
#         # Define figure filename
#         if prefix != "":
#             prefix = prefix + "_"
#         if suffix != "":
#             suffix = "_" + suffix
#         leadtime_str = "{:02d}".format((int(leadtime / np.timedelta64(1, "m"))))
#         fname = prefix + "L" + leadtime_str + suffix + ".png"
#         ##--------------------------------------------------------------------.
#         # Save figure
#         fig.savefig((figs_dir / fname), bbox_inches="tight")
#         ##--------------------------------------------------------------------.


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
    figsize=(17, 19)
):
    if not n_leadtimes:
        n_leadtimes = len(ds_averaged_skill.leadtime)
    # Plot first n_leadtimes
    ds_averaged_skill = ds_averaged_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_averaged_skill["leadtime"].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[m]")]
    # Create figure
    fig, axs = plt.subplots(len(skills), len(variables), figsize=figsize)
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


def plot_comparison_averaged_skills(
    list_ds_averaged_skill,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2", "KGE", "error_CoV"],
    variables=["precip"],
    legend_labels=None,
    n_leadtimes=None,
    figsize=(17, 19)
):
    if not n_leadtimes:
        n_leadtimes = len(list_ds_averaged_skill[0].leadtime)
    # Plot first n_leadtimes
    list_ds_averaged_skill = [ds.isel(leadtime=slice(0, n_leadtimes)) for 
                                ds in list_ds_averaged_skill]
    # Retrieve leadtime
    leadtimes = list_ds_averaged_skill[0]["leadtime"].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[m]")]
    # Create figure
    fig, axs = plt.subplots(len(skills), len(variables), figsize=figsize)
    # Initialize axes
    ax_i = 0
    axs = axs.flatten()
    for skill in skills:
        for var in variables:
            for ds_averaged_skill in list_ds_averaged_skill:
                # Plot global average skill
                axs[ax_i].plot(leadtimes, ds_averaged_skill[var].sel(skill=skill).values)
            
            if legend_labels:
                axs[ax_i].legend(legend_labels)
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