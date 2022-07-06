# %load_ext autoreload
# %autoreload 2

from pathlib import Path
from typing import List
import numpy as np
import xarray as xr
import pandas as pd
from tabulate import tabulate
from nowproject.config import create_test_events_autoregressive_time_range

from nowproject.data.data_utils import load_static_topo_data, prepare_data_dynamic, xr_sel_coords_between
from nowproject.visualization.plot_map import FixPointNormalize, plot_obs, plot_stats_map, plot_patches,\
                                        rescale_spatial_axes, plot_forecast_comparison, plot_forecasts_grid
from nowproject.visualization.plot_skills import plot_averaged_skill, plot_comparison_averaged_skills, plot_averaged_skills
from nowproject.data.dataset.data_config import METADATA_CH, METADATA
from pysteps.visualization.utils import proj4_to_cartopy

import matplotlib.pyplot as plt
import matplotlib.colors

from nowproject.visualization.plot_precip import _plot_map_cartopy

import seaborn as sns
sns.set_theme()

thresholds = [0.1, 5, 10]
spatial_scales = [5]

# Functions to load models skills and forecasts and show results

def load_models_skills_and_forecasts(models: List[str], models_dir: Path):
    cont = []
    cat = {thr: [] for thr in thresholds}
    spatial = {thr: [] for thr in thresholds}
    forecasts = []

    for model in models:
        experiment = models_dir / model / "model_skills"
        cont.append(xr.open_dataset(experiment / "deterministic_continuous_global_skill.nc"))
        for thr in thresholds:
            cat[thr].append(xr.open_dataset(experiment / f"deterministic_categorical_global_skill_thr{thr}_mean.nc"))
            for scale in spatial_scales:
                spatial[thr].append(xr.open_dataset(experiment / f"deterministic_spatial_global_skill_thr{thr}_scale{scale}.nc"))

        forecast_zarr_fpath = models_dir / model / "model_predictions" / "forecast_chunked" / "test_forecasts.zarr"
        forecasts.append(xr.open_zarr(forecast_zarr_fpath))
    
    return cont, cat, spatial, forecasts


def display_results(list_ds_skills: List[xr.Dataset], skills: List[str], labels_rows: List[str]):
    table = []
    leadtimes = list_ds_skills[0].leadtime.values
    leadtimes = [leadtimes[i] for i in [0, int(len(leadtimes)/2-1), int(len(leadtimes)-1)]]
    for ds_skills in list_ds_skills:
        results = []
        for skill in skills:
            results.extend(ds_skills.sel(skill=skill, leadtime=leadtimes).feature.round(3).values.tolist())
        table.append(results)
    
    table = pd.DataFrame(table, index=labels_rows)
    table.index.names = ['model']
    table.columns.names= ['metric']

   
    iterable = [skills, [str(l.astype("timedelta64[m]")).split(" ")[0] for l in leadtimes]]
    table.columns = pd.MultiIndex.from_product(iterable, names= ['group', 'subgroup'])

    h = [table.index.names[0] +'/'+ table.columns.names[0]] + list(map('\n'.join, table.columns.tolist()))
    print(tabulate(table, headers= h, tablefmt= 'fancy_grid'))

    print("\nTabulate Latex:")
    print(tabulate(table, headers=h, tablefmt="latex"))


def plot_forecasts_and_skills(forecasts, data_dynamic, legend_labels, forecasts_figs_dir, suptitle_prefix, 
                              filename_prefix, cont, cat, spatial, skills_figs_dir, geodata=METADATA_CH):
    for timestep in timesteps:
        list_forecasts = [ds.sel(forecast_reference_time=np.datetime64(timestep)) for ds in forecasts]                        

        plot_forecasts_grid(forecasts_figs_dir,
                            list_forecasts, data_dynamic,
                            legend_labels,
                            forecasts[0].leadtime.values[[0, 5, 11]],
                            geodata, 
                            suptitle_prefix=suptitle_prefix,
                            aspect_cbar=40,
                            filename_prefix=filename_prefix)

    display_results(cont, ["RMSE", "BIAS"], legend_labels)

    for thr in thresholds:
        print("Categorical metrics for threshold", thr)
        display_results(cat[thr], ["POD", "CSI", "F1"], legend_labels)
        print("Spatial metrics for threshold", thr)
        display_results(spatial[thr], ["SSIM", "FSS"], legend_labels)

    skills_figs_dir.mkdir(exist_ok=True)

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


default_data_dir = "/ltenas3/0_Data/NowProject/"
figs_dir = Path("/home/haddad/thesis_figs/")
data_dir_path  = Path(default_data_dir)
models_dir = data_dir_path / "experiments"
results_figs_dir = figs_dir / "results"

boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
data_dynamic_ch = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                        boundaries=boundaries, 
                                        timestep=5)

data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                    timestep=5)

test_events_path = Path("/home/haddad/nowproject/configs/subset_test_events.json")
test_events = create_test_events_autoregressive_time_range(test_events_path, 4, 
                                                               freq="5min")


timesteps = [
    "2016-04-16 18:00:00",
    "2017-01-12 17:00:00",
    "2017-01-31 16:00:00",
    "2017-06-14 16:00:00",
    # "2017-07-07 16:00:00",
    "2017-07-07 18:00:00",
    "2017-08-31 17:00:00"
]


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

mean = data_patches[["Min", "Max", "Mean", "Sum", "Area >= 1", "Area >= 5", "Area >= 20", 
                    "Dry-Wet Area Ratio"]].mean().to_frame().T.round(2)
median = data_patches[["Min", "Max", "Mean", "Sum", "Area >= 1", "Area >= 5", "Area >= 20", 
                    "Dry-Wet Area Ratio"]].median().to_frame().T.round(2)
q_25 = data_patches[["Min", "Max", "Mean", "Sum", "Area >= 1", "Area >= 5", "Area >= 20", 
                    "Dry-Wet Area Ratio"]].quantile(0.25).to_frame().T.round(2)
q_75 = data_patches[["Min", "Max", "Mean", "Sum", "Area >= 1", "Area >= 5", "Area >= 20", 
                    "Dry-Wet Area Ratio"]].quantile(0.75).to_frame().T.round(2)
q_95 = data_patches[["Min", "Max", "Mean", "Sum", "Area >= 1", "Area >= 5", "Area >= 20", 
                    "Dry-Wet Area Ratio"]].quantile(0.95).to_frame().T.round(2)
min = data_patches[["Min", "Max", "Mean", "Sum", "Area >= 1", "Area >= 5", "Area >= 20", 
                    "Dry-Wet Area Ratio"]].min().to_frame().T.round(2)
max = data_patches[["Min", "Max", "Mean", "Sum", "Area >= 1", "Area >= 5", "Area >= 20", 
                    "Dry-Wet Area Ratio"]].max().to_frame().T.round(2)

combined = pd.concat([min, q_25, median, mean, q_75, q_95, max], ignore_index=False)
combined.index = ["Min", "Q25", "Median", "Mean", "Q75", "Q95", "Max"]
combined


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


##------------------------------------------------------------------------.
## - Experiments
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

## - Compare scalers
models = [
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-4epochs-1year/",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-NormalizeScaler-MSEMasked-4epochs-1year",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-BinScaler-MSEMasked-4epochs-1year/"
]

cont, cat, spatial, forecasts = load_models_skills_and_forecasts(models, models_dir)
legend_labels = ["Log Normalizer", "Normalizer", "Bin"]

plot_forecasts_and_skills(forecasts, data_dynamic_ch, legend_labels, 
                          results_figs_dir / "comparison_forecasts", 
                          "Forecast comparison of a UNet3D trained with data scaled with different scalers", 
                          "scalers_", cont, cat, spatial, 
                          figs_dir / "results" / "skills_scalers", 
                          geodata=METADATA_CH)

## - Compare losses
models = [
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-4epochs-1year/",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-LogCoshMasked-4epochs-1year/",
]
cont, cat, spatial, forecasts = load_models_skills_and_forecasts(models, models_dir)
legend_labels = ["MSE Masked", "LogCosh Masked"]

plot_forecasts_and_skills(forecasts, data_dynamic_ch, legend_labels, 
                          results_figs_dir / "comparison_forecasts", 
                          "Forecast comparison of a UNet3D trained with MSE and LogCosh losses", 
                          "losses_", cont, cat, spatial, 
                          figs_dir / "results" / "skills_losses", 
                          geodata=METADATA_CH)


## - Compare models
models = [
    "RNN-AR6-resConv64-IncrementLearning-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year",
    "RNN-AR6-ResidualUNet3D-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year",
    "RNN-AR6-MultiScaleResidualConv-IncrementLearning-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year"
]
cont, cat, spatial, forecasts = load_models_skills_and_forecasts(models, models_dir)
legend_labels = ["resConv64", "UNet3D-ReLU", "UNet3D-ELU", "MultiScaleResConv-ELU"]

plot_forecasts_and_skills(forecasts, data_dynamic_ch, legend_labels, 
                          results_figs_dir / "comparison_forecasts", 
                          "Forecast comparison of the different implemented models",
                          "models_", cont, cat, spatial, 
                          figs_dir / "results" / "skills_models", 
                          geodata=METADATA_CH)

# Compare MSE non-weighted vs weighted
models = [
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMaskedWeightedb5c1-15epochs-1year",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMaskedWeightedb5c4-15epochs-1year"
]
cont, cat, spatial, forecasts = load_models_skills_and_forecasts(models, models_dir)

legend_labels = ["No Weighting", "Weighedtb5c1", "Weighedtb5c4"]

plot_forecasts_and_skills(forecasts, data_dynamic_ch, legend_labels, 
                          results_figs_dir / "comparison_forecasts", 
                          "Forecast comparison of model trained with non-weighted vs weighted MSE",
                          "weighted_mse_", cont, cat, spatial, 
                          figs_dir / "results" / "skills_weighted_mse", 
                          geodata=METADATA_CH)


# Adding more training data
models = [
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-2years"
]
cont, cat, spatial, forecasts = load_models_skills_and_forecasts(models, models_dir)
legend_labels = ["1 year\n(2018)", "2 years\n(2018-2019)"]

plot_forecasts_and_skills(forecasts, data_dynamic_ch, legend_labels, 
                          results_figs_dir / "comparison_forecasts", 
                          "Forecast comparison of model trained with 1 year vs 2 years of data",
                          "more_training_data_", cont, cat, spatial, 
                          figs_dir / "results" / "skills_more_data", 
                          geodata=METADATA_CH)

# No static feature vs DEM
models = [
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-DEM-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year"
]
cont, cat, spatial, forecasts = load_models_skills_and_forecasts(models, models_dir)
legend_labels = ["No static feature", "DEM"]

plot_forecasts_and_skills(forecasts, data_dynamic_ch, legend_labels, 
                          results_figs_dir / "comparison_forecasts", 
                          "Forecast comparison of model trained with no static feature and with DEM",
                          "static_features_", cont, cat, spatial, 
                          figs_dir / "results" / "skills_static_features", 
                          geodata=METADATA_CH)

# Plot best models
models = [
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMaskedWeightedb5c4-15epochs-1year"
]
cont, cat, spatial, forecasts = load_models_skills_and_forecasts(models, models_dir)
legend_labels = ["UNet3D-ELU\nMSEMasked\nNonWeighted", "UNet3D-ELU\nMSEMasked\nWeightedb5c4"]

plot_forecasts_and_skills(forecasts, data_dynamic_ch, legend_labels, 
                          results_figs_dir / "comparison_forecasts", 
                          "Forecast comparison of the best models",
                          "best_models_", cont, cat, spatial, 
                          figs_dir / "results" / "skills_best_models", 
                          geodata=METADATA_CH)



# Compare best model vs benchmarks

extrapolation_dir_path = data_dir_path / "extrapolation" / "5min_full"
benchmark_dir_path = data_dir_path / "benchmarks" / "5min_full"

list_ds_lk = []
list_ds_benchmark = []
for idx, event in enumerate(test_events):
    test_event_dir = extrapolation_dir_path / f"test_event_{idx}"
    list_ds_lk.append(xr.open_zarr(test_event_dir / f"extrapolation_test_event_{idx}.zarr"))
    test_event_dir = benchmark_dir_path / f"test_event_{idx}"
    list_ds_benchmark.append(xr.open_zarr(test_event_dir / f"benchmark_test_event_{idx}.zarr"))

ds_lk = xr.concat(list_ds_lk, dim="forecast_reference_time")
ds_lk = xr_sel_coords_between(ds_lk, **boundaries)
ds_lk = ds_lk.rename({"lucas_kanade": "feature"})[["feature"]]

ds_benchmark = xr.concat(list_ds_benchmark, dim="forecast_reference_time")
ds_benchmark = xr_sel_coords_between(ds_benchmark, **boundaries)

benchmark_skills_dir = benchmark_dir_path / "combined_ch" / "skills"
models_dir = data_dir_path / "experiments"

cont = []
cat = {thr: [] for thr in thresholds}
spatial = {thr: [] for thr in thresholds}
for key in (ds_benchmark.data_vars.keys()):
    skills_dir = (benchmark_skills_dir / key)
    cont.append(xr.open_dataset(skills_dir / "deterministic_continuous_global_skill.nc"))
    for thr in thresholds:
        cat[thr].append(xr.open_dataset(skills_dir / f"deterministic_categorical_global_skill_thr{thr}_mean.nc"))
        for scale in spatial_scales:
            spatial[thr].append(xr.open_dataset(skills_dir / f"deterministic_spatial_global_skill_thr{thr}_scale{scale}.nc"))

extrapolation_skills_dir = extrapolation_dir_path / "combined_ch" / "skills"

skills_dir = (extrapolation_skills_dir / "lucas_kanade")
cont.append(xr.open_dataset(skills_dir / "deterministic_continuous_global_skill.nc"))
for thr in thresholds:
    cat[thr].append(xr.open_dataset(skills_dir / f"deterministic_categorical_global_skill_thr{thr}_mean.nc"))
    for scale in spatial_scales:
        spatial[thr].append(xr.open_dataset(skills_dir / f"deterministic_spatial_global_skill_thr{thr}_scale{scale}.nc"))



forecasts = [ds_benchmark.rename({var: "feature"})[["feature"]] \
                        for var in list(ds_benchmark.data_vars.keys())]
forecasts += [ds_lk]

models = [
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMasked-15epochs-1year",
    "RNN-AR6-ResidualUNet3D-ELU-IncrementLearning-NoAct-ReZero-5mins-Patches-LogNormalizeScaler-MSEMaskedWeightedb5c4-15epochs-1year"
]

for model in models:
    experiment = models_dir / model / "model_skills"
    cont.append(xr.open_dataset(experiment / "deterministic_continuous_global_skill.nc"))
    for thr in thresholds:
        cat[thr].append(xr.open_dataset(experiment / f"deterministic_categorical_global_skill_thr{thr}_mean.nc"))
        for scale in spatial_scales:
            spatial[thr].append(xr.open_dataset(experiment / f"deterministic_spatial_global_skill_thr{thr}_scale{scale}.nc"))

    forecast_zarr_fpath = models_dir / model / "model_predictions" / "forecast_chunked" / "test_forecasts.zarr"
    forecasts.append(xr.open_zarr(forecast_zarr_fpath))


legend_labels = ["S-PROG", "STEPS Mean", "STEPS Median", "Extrapolation", 
                    "UNet3D-ELU\nMSEMasked\nNonWeighted", "UNet3D-ELU\nMSEMasked\nWeightedb5c4"]

plot_forecasts_and_skills(forecasts, data_dynamic_ch, legend_labels, 
                          results_figs_dir / "comparison_forecasts", 
                          "Forecast comparison of 2 UNet3D models and the benchmarks",
                          "benchmarks_vs_best_", cont, cat, spatial, 
                          figs_dir / "results" / "skills_benchmarks_vs_best", 
                          geodata=METADATA_CH)


# Optical flow comparison
extrapolation_dir_path = data_dir_path / "extrapolation" / "5min_full"
list_ds_lk = []
list_ds_raft = []

cont = []
cat = {thr: [] for thr in thresholds}
spatial = {thr: [] for thr in thresholds}

for idx, event in enumerate(test_events):
    test_event_dir = extrapolation_dir_path / f"test_event_{idx}"
    list_ds_lk.append(xr.open_zarr(test_event_dir / f"extrapolation_test_event_{idx}.zarr"))
    list_ds_raft.append(xr.open_zarr(test_event_dir / f"raft_test_event_{idx}.zarr"))
    for key in ["lucas_kanade", "raft"]:
        skills_dir = (extrapolation_skills_dir / key)
        cont.append(xr.open_dataset(skills_dir / "deterministic_continuous_global_skill.nc"))
        for thr in thresholds:
            cat[thr].append(xr.open_dataset(skills_dir / f"deterministic_categorical_global_skill_thr{thr}_mean.nc"))
            for scale in spatial_scales:
                spatial[thr].append(xr.open_dataset(skills_dir / f"deterministic_spatial_global_skill_thr{thr}_scale{scale}.nc"))




ds_lk = xr.concat(list_ds_lk, dim="forecast_reference_time")
ds_lk = xr_sel_coords_between(ds_lk, **boundaries)
ds_lk = ds_lk.rename({"lucas_kanade": "feature"})[["feature"]]
ds_raft = xr.concat(list_ds_raft, dim="forecast_reference_time")
ds_raft = xr_sel_coords_between(ds_raft, **boundaries)
ds_raft = ds_raft.rename({"raft": "feature"})[["feature"]]

forecasts = [ds_lk, ds_raft]
legend_labels = ["Lucas-Kanade", "RAFT"]

plot_forecasts_and_skills(forecasts, data_dynamic_ch, legend_labels, 
                          results_figs_dir / "comparison_forecasts", 
                          "Forecast comparison of optical flows with extrapolation",
                          "extrapolation_", cont, cat, spatial, 
                          figs_dir / "results" / "skills_extrapolation", 
                          geodata=METADATA_CH)
