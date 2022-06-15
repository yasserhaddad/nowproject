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

import seaborn as sns
sns.set_theme()

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


def plot_averaged_skill(
    ds_averaged_skill, skill="RMSE", variables=["precip"], n_leadtimes=None,
    title=None
):
    if not n_leadtimes:
        n_leadtimes = len(ds_averaged_skill.leadtime)
    # Plot first n_leadtimes
    ds_averaged_skill = ds_averaged_skill.isel(leadtime=slice(0, n_leadtimes))
    # Retrieve leadtime
    leadtimes = ds_averaged_skill["leadtime"].values
    leadtimes = [str(l).split(" ")[0] for l in leadtimes.astype("timedelta64[m]")]
    # Create figure
    fig, axs = plt.subplots(1, len(variables), figsize=(12, 4))
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
        if title:
            ax.set_title(title)
        else:
            ax.set_title(var.upper())
        ##------------------------------------------------------------------.
    fig.tight_layout()
    return fig


def plot_averaged_skills(
    ds_averaged_skill,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2", "KGE", "error_CoV"],
    variables=["precip"],
    n_leadtimes=None,
    figsize=(12, 19)
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
    fig, axs = plt.subplots(len(skills), len(variables), figsize=(12, 18))
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