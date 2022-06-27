# %load_ext autoreload
# %autoreload 2

from pathlib import Path
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr
import pandas as pd

from nowproject.data.data_utils import load_static_topo_data, prepare_data_dynamic, xr_sel_coords_between
from nowproject.utils.plot_map import FixPointNormalize, plot_obs, plot_stats_map, plot_patches,\
                                        rescale_spatial_axes, plot_forecast_comparison
from nowproject.utils.plot_skills import plot_averaged_skill, plot_comparison_averaged_skills, plot_averaged_skills
from nowproject.data.data_config import METADATA_CH, METADATA
from pysteps.visualization.utils import proj4_to_cartopy

import matplotlib.pyplot as plt
import matplotlib.colors

import seaborn as sns

from nowproject.utils.plot_precip import _plot_map_cartopy
sns.set_theme()

default_data_dir = "/ltenas3/0_Data/NowProject/"
figs_dir = Path("/home/haddad/thesis_figs/")
data_dir_path  = Path(default_data_dir)

boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
data_dynamic_ch = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                        boundaries=boundaries, 
                                        timestep=5)
data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                    timestep=5)

# Example of observation
time_start = np.datetime64('2021-06-22T17:35:00.000000000')
time_end =  np.datetime64('2021-06-22T17:45:00.000000000')
plot_obs(figs_dir / "full", data_dynamic.sel(time=slice(time_start, time_end)), METADATA, save_gif=False, figsize=(12, 7))

time_start = np.datetime64('2021-06-22T17:35:00.000000000')
time_end =  np.datetime64('2021-06-22T17:45:00.000000000')
plot_obs(figs_dir / "swiss", data_dynamic_ch.sel(time=slice(time_start, time_end)), METADATA_CH, save_gif=False, figsize=(12, 7))

# Data statistics

(figs_dir / "data_stats").mkdir(exist_ok=True)

data_stats_dir_path = data_dir_path / "stats_5min"

## Counts
df_counts = pd.read_csv(data_stats_dir_path / "counts_values.csv")
df_counts = df_counts.groupby("value").sum().reset_index()

print("Percentage of 0s in dataset: ", 
        np.round(df_counts[df_counts.value == 0]["count"].values[0] / sum(df_counts["count"]) * 100, 1))

print("Percentage of values above 60: ",
        np.round(sum(df_counts[df_counts.value >= 60]["count"].values) / sum(df_counts["count"]) * 100, 5)) 

# bins = [0.0, 0.1, 0.4, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0, 120.0]
bins = [0.0, 0.1, 0.6, 1, 1.6, 2.5, 5, 10, 20, 40, 65, 100, 120]
fig = plt.figure(dpi=300)
df_counts.hist(column="value", weights=df_counts["count"], bins=bins, figsize=(12, 5), linewidth=0.3)
plt.xlabel("Precipitation intensity (mm/h)")
plt.yscale("log")
plt.ylabel("Log-count")
plt.title("Histogram of the distribution of the log-counts of precipitation intensities in the dataset", y=1.03)
plt.savefig(figs_dir / "data_stats" / "dist_log_counts.png", dpi=300, bbox_inches='tight')


mean_space_month = xr.open_dataset(data_stats_dir_path / "stats_space_month.nc")
plot_stats_map(figs_dir / "data_stats" / "month", mean_space_month, METADATA_CH, 
                figsize=(12, 7), time_coord="month", title_prefix="")

mean_month = xr.open_dataset(data_stats_dir_path / "stats_month.nc")

fig = plt.figure(figsize=(12, 7))
plt.bar(mean_month.month, mean_month["mean"])
plt.xticks(mean_month.month)
plt.xlabel("Month")
plt.ylabel("Mean precipitation intensity (mm/h)")
plt.title("Mean precipitation intensity in Switzerland per month")
plt.savefig(figs_dir / "data_stats" / "mean_precip_per_month.png", dpi=300, bbox_inches='tight')


mean_year = xr.open_dataset(data_stats_dir_path / "stats_year.nc")

fig = plt.figure(figsize=(8, 5))
plt.bar(mean_year.year, mean_year["mean"], width=0.7)
plt.xticks(mean_year.year)
plt.ylim(top=0.20)
plt.xlabel("Year")
plt.ylabel("Mean precipitation intensity (mm/h)")
plt.title("Mean precipitation intensity in Switzerland per year")
plt.savefig(figs_dir / "data_stats" / "mean_precip_per_year.png", dpi=300, bbox_inches='tight')


# Example of precipitation patch
(figs_dir / "patches").mkdir(exist_ok=True)
plot_patches(figs_dir / "patches", data_dynamic_ch.feature.sel(time="2018-03-22T15:30:00"))

data_patches = pd.read_parquet(data_dir_path / "rzc_cropped_patches_fixed.parquet")
data_patches = data_patches[data_patches.time.dt.minute % 5 == 0]


## Statistics

print("Timesteps coverage", np.round(len(np.unique(data_patches.time)) / len(data_dynamic.time) * 100, 2))

mean = data_patches[["Max", "Min", "Mean", "Area >= 1", "Area >= 5", "Area >= 20", "Sum", 
                    "Dry-Wet Area Ratio"]].mean().to_frame().T.round(2)
median = data_patches[["Max", "Min", "Mean", "Area >= 1", "Area >= 5", "Area >= 20", "Sum", 
                    "Dry-Wet Area Ratio"]].median().to_frame().T.round(2)



# Topographic data
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
colors = np.vstack((colors_undersea, colors_land))
cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors)

dem_dir_path = Path("/ltenas3/0_GIS/DEM Switzerland/")
dem = rescale_spatial_axes(load_static_topo_data(dem_dir_path / "srtm_Switzerland_EPSG21781.tif", 
                                                 data_dynamic_ch, upsample=False))


norm = FixPointNormalize(sealevel=0,
                         vmax=np.max(dem.feature.data),
                         vmin=np.min(dem.feature.data))
crs_ref = proj4_to_cartopy(METADATA_CH["projection"])
crs_proj = crs_ref
_, ax = plt.subplots(
    figsize=(8, 5),
    subplot_kw={'projection': crs_proj}
)

cbar_kwargs = {
    "spacing": "uniform",
    "extend": "neither",
    "shrink": 0.8,
    "label": "Elevation (m)"
}

p = dem.feature.plot.imshow(
        ax=ax,
        transform=crs_ref,
        norm=norm,
        cmap=cut_terrain_map, 
        interpolation="nearest",
        add_colorbar=True,
        cbar_kwargs=cbar_kwargs,
        zorder=1,
    )
ax.set_title("Topographic map of Switzerland")
p.axes = _plot_map_cartopy(crs_proj, 
                            cartopy_scale="50m",
                            drawlonlatlines=False,
                            ax=p.axes,
                            lw=1.0)
plt.savefig(figs_dir / "topographic_map.png", dpi=300, bbox_inches='tight')


# Results

results_figs_dir = figs_dir / "results"
results_figs_dir.mkdir(exist_ok=True)

benchmark_dir_path = data_dir_path / "benchmarks" / "5min_full"
# Test event 1
timestep = np.datetime64("2016-04-16 18:00:00")
test_event_dir = benchmark_dir_path / "test_event_0"
ds_forecast_benchmark = xr.open_zarr(test_event_dir / "benchmark_test_event_0.zarr")
ds_forecast_benchmark = xr_sel_coords_between(ds_forecast_benchmark, **boundaries)

# Test event 2
timestep = np.datetime64("2017-01-12 17:00:00")
test_event_dir = benchmark_dir_path / "test_event_1"
ds_forecast_benchmark = xr.open_zarr(test_event_dir / "benchmark_test_event_1.zarr")
ds_forecast_benchmark = xr_sel_coords_between(ds_forecast_benchmark, **boundaries)

## SPROG and STEPS
ds_forecast = ds_forecast_benchmark.sel(forecast_reference_time=timestep)\
                                    .rename({"sprog": "feature"})[["feature"]]

plot_forecast_comparison(results_figs_dir / "benchmark_sprog", 
                            ds_forecast,
                            data_dynamic_ch,
                            geodata=METADATA_CH,
                            suptitle_prefix="S-PROG, ")

ds_forecast = ds_forecast_benchmark.sel(forecast_reference_time=timestep)\
                                    .rename({"steps_mean": "feature"})[["feature"]]

plot_forecast_comparison(results_figs_dir / "benchmark_steps_mean", 
                            ds_forecast,
                            data_dynamic_ch,
                            geodata=METADATA_CH,
                            suptitle_prefix="STEPS Mean, ")

ds_forecast = ds_forecast_benchmark.sel(forecast_reference_time=timestep)\
                                    .rename({"steps_median": "feature"})[["feature"]]

plot_forecast_comparison(results_figs_dir / "benchmark_steps_median", 
                            ds_forecast,
                            data_dynamic_ch,
                            geodata=METADATA_CH,
                            suptitle_prefix="STEPS Median, ")



# Compare skills
benchmark_skills_dir = benchmark_dir_path / "combined_ch" / "skills"
models_dir = data_dir_path / "experiments"

# for key in (ds_forecast_benchmark.data_vars.keys()):
#     skills_dir = (benchmark_skills_dir / key)
#     ds_cont_averaged_skill = xr.open_dataset(skills_dir / "deterministic_continuous_global_skill.nc")
#     ds_cat_averaged_skill = xr.open_dataset(skills_dir / "deterministic_categorical_global_skill.nc")
#     ds_spatial_average_skill = xr.open_dataset(skills_dir / "deterministic_spatial_global_skill.nc")

#     (skills_dir / "figs").mkdir(exist_ok=True)
#     plot_averaged_skill(ds_cont_averaged_skill, skill="RMSE", variables=["feature"]).savefig(
#         skills_dir / "figs" / "RMSE_skill.png"
#     )
#     plot_averaged_skills(ds_cont_averaged_skill, 
#                         skills=["BIAS", "RMSE"], 
#                         variables=["feature"], figsize=(15, 8)).savefig(
#         skills_dir / "figs" / "averaged_continuous_skill.png"
#     )

#     plot_averaged_skills(ds_cat_averaged_skill, 
#                         skills=["POD", "CSI", "F1"], variables=["feature"],
#                         figsize=(15, 12)).savefig(
#         skills_dir / "figs" / "averaged_categorical_skills.png"
#     )

#     plot_averaged_skills(ds_spatial_average_skill, 
#                         skills=["SSIM", "FSS"], variables=["feature"], figsize=(15, 8)).savefig(
#         skills_dir / "figs" / "averaged_spatial_skills.png"
#     )

thresholds = [0.1, 5, 10]
spatial_scales = [5]

cont = []
cat = {thr: [] for thr in thresholds}
spatial = {thr: [] for thr in thresholds}
for key in (ds_forecast_benchmark.data_vars.keys()):
    skills_dir = (benchmark_skills_dir / key)
    cont.append(xr.open_dataset(skills_dir / "deterministic_continuous_global_skill.nc"))
    for thr in thresholds:
        cat[thr].append(xr.open_dataset(skills_dir / f"deterministic_categorical_global_skill_thr{thr}_mean.nc"))
        for scale in spatial_scales:
            spatial[thr].append(xr.open_dataset(skills_dir / f"deterministic_spatial_global_skill_thr{thr}_scale{scale}.nc"))

for experiment in models_dir.glob("*/model_skills/"):
    cont.append(xr.open_dataset(experiment / "deterministic_continuous_global_skill.nc"))
    for thr in thresholds:
        cat[thr].append(xr.open_dataset(experiment / f"deterministic_categorical_global_skill_thr{thr}_mean.nc"))
        for scale in spatial_scales:
            spatial[thr].append(xr.open_dataset(experiment / f"deterministic_spatial_global_skill_thr{thr}_scale{scale}.nc"))



skills_figs_dir = figs_dir / "results" / "skills"
skills_figs_dir.mkdir(exist_ok=True)

legend_labels = ["S-PROG", "STEPS Mean", "STEPS Median", "UNet3D-MSEMasked", "UNet3D-MSEMaskedWeighted"]

plot_comparison_averaged_skills(cont, 
                    skills=["BIAS", "RMSE"], variables=["feature"], 
                    legend_labels=legend_labels, title=f"Continuous Metrics",
                    figsize=(12, 8)).savefig(
    skills_figs_dir / "averaged_continuous_skill.png"
)

for thr in thresholds:
    plot_comparison_averaged_skills(cat[thr], 
                        skills=["POD", "CSI", "F1"], variables=["feature"],
                        legend_labels=legend_labels, title=f"Categorical Metrics, Threshold {thr}",
                        figsize=(12, 12)).savefig(
        skills_figs_dir / f"averaged_categorical_skills_thr{thr}.png"
    )

    plot_comparison_averaged_skills(spatial[thr], legend_labels=legend_labels,
                        skills=["SSIM", "FSS"], variables=["feature"], title=f"Spatial Metrics, Threshold {thr}",
                        figsize=(12, 8)).savefig(
        skills_figs_dir / f"averaged_spatial_skills_thr{thr}.png"
    )

