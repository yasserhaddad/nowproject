import xarray as xr
import numpy as np

from pathlib import Path
from pysteps import motion, nowcasts

from nowproject.utils.config import (
    create_test_events_autoregressive_time_range
)
from nowproject.data.data_utils import prepare_data_dynamic
from xforecasting.predictions_autoregressive import reshape_forecasts_for_verification

LEADTIME = 21
FREQ_IN_NS = 150000000000
LEADTIMES = np.arange(FREQ_IN_NS, FREQ_IN_NS*(LEADTIME + 1), FREQ_IN_NS, dtype="timedelta64[ns]")
AR = [-3, -2, -1]

data_dir_path = Path("/ltenas3/0_Data/NowProject/")
test_events_path = Path("/home/haddad/nowproject/configs/events.json")
exp_dir_path = Path("/home/haddad/experiments/")

data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr")
test_events = create_test_events_autoregressive_time_range(test_events_path, len(AR))

sprog = []
steps = []
for event in test_events:
    event_sprog = []
    event_steps = []
    oflow_method = motion.get_method("lucaskanade")
    R = data_dynamic.sel(time=event).feature.values
    V = oflow_method(R)
    for i in range(len(event) - len(AR)):
        nowcast_method = nowcasts.get_method("sprog")
        R_f_sprog = nowcast_method(R[i:i + len(AR) + 1, :, :], V, timesteps=LEADTIME, R_thr=0.1)
        event_sprog.append(R_f_sprog)

        nowcast_method = nowcasts.get_method("steps")
        R_f_steps = nowcast_method(R[i:i + len(AR) + 1, :, :], V, timesteps=LEADTIME, 
                                   n_ens_members=24, n_cascade_levels=8, kmperpixel=1.0, 
                                   R_thr=0.1, timestep=2.5)

        R_f_steps = np.mean(R_f_steps, axis=0)
        event_steps.append(R_f_steps)

    sprog.append(np.asarray(event_sprog))
    steps.append(np.asarray(event_steps))

ds_benchmark = xr.Dataset(
    data_vars=dict(
        sprog=(["forecast_reference_time", "leadtime", "y", "x"], np.concatenate(sprog, axis=0)),
        steps=(["forecast_reference_time", "leadtime", "y", "x"], np.concatenate(steps, axis=0)),
    ),
    coords=dict(
        forecast_reference_time=np.concatenate([event[len(AR):] for event in test_events]),
        leadtime=LEADTIMES,
        x=(["x"], data_dynamic.x.data),
        y=(["y"], data_dynamic.y.data),
    ),
    attrs={},
)

ds_benchmark = reshape_forecasts_for_verification(ds_benchmark)

ds_benchmark.to_zarr(exp_dir_path / "benchmark.zarr", mode="w")

