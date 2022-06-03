import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
from pysteps.visualization.utils import proj4_to_cartopy

from nowproject.data.data_utils import prepare_data_dynamic, load_static_topo_data
from nowproject.data.data_config import METADATA_CH
from nowproject.utils.plot_precip import plot_single_precip, _plot_map_cartopy
from nowproject.utils.plot_map import plot_forecast_comparison

from nowproject.utils.plot import (
    plot_averaged_skill,
    plot_averaged_skills, 
    plot_comparison_averaged_skills
)

# %load_ext autoreload
# %autoreload 2

data_dir_path = Path("/ltenas3/0_Data/NowProject/")
figs_dir_path = Path("/home/haddad/presentation_figs/")


data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr")

time_1 = np.datetime64('2021-06-28T15:30:00.000000000')
time_2 = np.datetime64('2021-06-22T17:40:00.000000000')

boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
data_dynamic_ch = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr",
                                       boundaries=boundaries, timestep=5)


# Data statistics
data_stats_dir_path = data_dir_path / "stats"
value_counts = pd.read_csv(data_stats_dir_path / "counts_values.csv")

ds_stats_space = xr.open_dataset(data_stats_dir_path / "stats_space.nc")

ds_stats_space_month = xr.open_dataset(data_stats_dir_path / "stats_space_month.nc")

# Data patches
data_patches = pd.read_parquet(data_dir_path / "rzc_cropped_patches.parquet")

# Topographic data

## Fix for terrain colormap 
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

# Combine the lower and upper range of the terrain colormap with a gap in the middle
# to let the coastline appear more prominently.
# inspired by https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
colors = np.vstack((colors_undersea, colors_land))
cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors)

dem_dir_path = Path("/ltenas3/0_GIS/DEM Switzerland/")
dem = load_static_topo_data(dem_dir_path / "srtm_Switzerland_EPSG21781.tif", data_dynamic_ch)
dem = dem.assign_coords({"x": dem.x.data*1000})
dem = dem.assign_coords({"y": dem.y.data*1000})

norm = FixPointNormalize(sealevel=0,
                         vmax=np.max(dem.feature.data)-400,
                         vmin=np.min(dem.feature.data)+250)
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
                            ax=p.axes)
plt.savefig(figs_dir_path / "topographic_map.png", dpi=300, bbox_inches='tight')


# Benchmark
timestep = np.datetime64("2016-04-16 18:00:00")
benchmark_dir_path = data_dir_path / "benchmarks"
test_event_dir = benchmark_dir_path / "test_event_0"
ds_forecast_benchmark1 = xr.open_zarr(test_event_dir / "benchmark_test_event_0.zarr")

ds_forecast = ds_forecast_benchmark1.sel(forecast_reference_time=timestep)\
                                    .rename({"sprog": "feature"})[["feature"]]

plot_forecast_comparison(figs_dir_path / "benchmark_sprog", 
                               ds_forecast,
                               data_dynamic_ch,
                               geodata=METADATA_CH,
                               suptitle_prefix="S-PROG, ")

ds_forecast = ds_forecast_benchmark1.sel(forecast_reference_time=timestep)\
                                    .rename({"steps": "feature"})[["feature"]]

plot_forecast_comparison(figs_dir_path / "benchmark_steps", 
                               ds_forecast,
                               data_dynamic_ch,
                               geodata=METADATA_CH,
                               suptitle_prefix="STEPS, ")

# Benchmark skills
for key in (ds_forecast_benchmark1.data_vars.keys()):
    skills_dir = (test_event_dir / "skills" / key)
    ds_cont_averaged_skill = xr.open_dataset(skills_dir / "deterministic_continuous_global_skill.nc")
    ds_cat_averaged_skill = xr.open_dataset(skills_dir / "deterministic_categorical_global_skill.nc")
    ds_spatial_average_skill = xr.open_dataset(skills_dir / "deterministic_spatial_global_skill.nc")

    (skills_dir / "figs").mkdir(exist_ok=True)
    plot_averaged_skill(ds_cont_averaged_skill, skill="RMSE", variables=["feature"]).savefig(
        skills_dir / "figs" / "RMSE_skill.png"
    )
    plot_averaged_skills(ds_cont_averaged_skill, 
                        skills=["BIAS", "RMSE"], 
                        variables=["feature"], figsize=(15, 8)).savefig(
        skills_dir / "figs" / "averaged_continuous_skill.png"
    )

    plot_averaged_skills(ds_cat_averaged_skill, 
                        skills=["POD", "CSI", "F1"], variables=["feature"],
                        figsize=(15, 12)).savefig(
        skills_dir / "figs" / "averaged_categorical_skills.png"
    )

    plot_averaged_skills(ds_spatial_average_skill, 
                        skills=["SSIM", "FSS"], variables=["feature"], figsize=(15, 8)).savefig(
        skills_dir / "figs" / "averaged_spatial_skills.png"
    )

cont = []
cat = []
spatial = []
for key in (ds_forecast_benchmark1.data_vars.keys()):
    skills_dir = (test_event_dir / "skills" / key)
    cont.append(xr.open_dataset(skills_dir / "deterministic_continuous_global_skill.nc"))
    cat.append(xr.open_dataset(skills_dir / "deterministic_categorical_global_skill.nc"))
    spatial.append(xr.open_dataset(skills_dir / "deterministic_spatial_global_skill.nc"))

(test_event_dir / "skills" / "comparison").mkdir(exist_ok=True)
plot_comparison_averaged_skills(cont, 
                    skills=["BIAS", "RMSE"], variables=["feature"], 
                    legend_labels=["S-PROG", "STEPS"], figsize=(15, 8)).savefig(
    test_event_dir / "skills" / "comparison" / "averaged_continuous_skill.png"
)

plot_comparison_averaged_skills(cat, 
                    skills=["POD", "CSI", "F1"], variables=["feature"],
                    legend_labels=["S-PROG", "STEPS"], figsize=(15, 12)).savefig(
    test_event_dir / "skills" / "comparison" / "averaged_categorical_skills.png"
)

plot_comparison_averaged_skills(spatial, legend_labels=["S-PROG", "STEPS"],
                    skills=["SSIM", "FSS"], variables=["feature"], figsize=(15, 8)).savefig(
    test_event_dir / "skills" / "comparison" / "averaged_spatial_skills.png"
)



