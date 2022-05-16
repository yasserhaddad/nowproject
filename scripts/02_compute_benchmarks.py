import xarray as xr
import numpy as np

from pathlib import Path
from pysteps import motion, nowcasts

from nowproject.utils.config import (
    create_test_events_autoregressive_time_range
)

data_dir_path = Path("/ltenas3/0_Data/NowProject/")
test_events_path = Path("/home/haddad/nowproject/configs/events.json")
exp_dir_path = Path("/home/haddad/experiments/")

data_dynamic = xr.open_zarr(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr")
data_dynamic = data_dynamic.reset_coords(
    ["radar_names", "radar_quality", "radar_availability"], 
    drop=True
    )
data_dynamic = data_dynamic.sel(time=slice(None, "2021-09-01T00:00"))
data_dynamic = data_dynamic.sel(
    {"y": list(range(835, 470, -1)), 
    "x": list(range(60, 300))}
    )
data_dynamic = data_dynamic.rename({"precip": "feature"})[["feature"]]

test_events = create_test_events_autoregressive_time_range(test_events_path, [-3, -2, -1])

sprog = []
steps = []
for event in test_events:
    oflow_method = motion.get_method("lucaskanade")
    R = data_dynamic.sel(time=event).feature.values
    V = oflow_method(R)

    nowcast_method = nowcasts.get_method("sprog")
    R_f_sprog = nowcast_method(R[:3, :, :], V, timesteps=len(event[2:]), R_thr=0.1)
    sprog.append(R_f_sprog)

    nowcast_method = nowcasts.get_method("steps")
    R_f_steps = nowcast_method(R[:3, :, :], V, timesteps=len(event[2:]), n_ens_members=24, 
                            n_cascade_levels=8, kmperpixel=1.0, R_thr=0.1, timestep=2.5)

    R_f_steps = np.mean(R_f_steps, axis=0)
    steps.append(R_f_steps)

ds_benchmark = xr.Dataset(
    data_vars=dict(
        sprog=(["time", "y", "x"], np.concatenate(sprog, axis=0)),
        steps=(["time", "y", "x"], np.concatenate(steps, axis=0)),
    ),
    coords=dict(
        time=np.concatenate([event[2:] for event in test_events]),
        x=(["x"], data_dynamic.x.data),
        y=(["y"], data_dynamic.y.data),
    ),
    attrs={},
)

ds_benchmark.to_zarr(exp_dir_path / "benchmark.zarr")

