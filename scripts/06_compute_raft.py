import time
import xarray as xr
import numpy as np
import torch

from pathlib import Path
from pysteps import motion, nowcasts

from nowproject.config import (
    create_test_events_autoregressive_time_range
)
from nowproject.data.data_utils import prepare_data_dynamic, xr_sel_coords_between
from xforecasting.predictions_autoregressive import reshape_forecasts_for_verification

from pysteps.utils import transformation
from nowproject.verification.verification import verification_routine

from nowproject.data.scalers_modules import log_normalize_scaler
# from nowproject.models.layers_optical_flow import RAFTOpticalFlow


NUM_WORKERS = 15
LEADTIME = 12
FREQ_IN_NS = 150000000000 * 2
LEADTIMES = np.arange(FREQ_IN_NS, FREQ_IN_NS*(LEADTIME + 1), FREQ_IN_NS, dtype="timedelta64[ns]")
AR = [-4, -3, -2, -1]


def compute_optical_flow(R, raft):
    batch_1 = torch.Tensor(np.expand_dims(R[:len(AR)], 1))
    batch_2 = torch.Tensor(np.expand_dims(R[1:len(AR) + 1], 1))
    flows, _, _, _ = raft.compute_flow_field(batch_1, batch_2)
    return flows.mean(axis=0)

if __name__ == "__main__":
    t_i = time.time()

    data_dir_path = Path("/ltenas3/0_Data/NowProject/")
    test_events_path = Path("/home/haddad/nowproject/configs/subset_test_events.json")

    boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
    timestep = 5
    data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr",
                                        timestep=timestep)
    scaler = log_normalize_scaler()
    test_events = create_test_events_autoregressive_time_range(test_events_path, len(AR), 
                                                               freq="5min")
    benchmark_dir_path = data_dir_path / "extrapolation" / "5min_full"
    benchmark_dir_path.mkdir(parents=True, exist_ok=True)
    raft = RAFTOpticalFlow(len(AR), small_model=True, finetune=False)

    for idx, event in enumerate(test_events):
        test_event_dir = benchmark_dir_path / f"test_event_{idx}"
        test_event_dir.mkdir(exist_ok=True, parents=True)

        event_raft = []

        R_scaled = scaler.transform(data_dynamic.sel(time=event)).feature

        R = data_dynamic.sel(time=event).feature.values
        R = transformation.dB_transform(R, threshold=0.1, zerovalue=-15.0)[0]
        R[~np.isfinite(R)] = -15.0

        for i in range(len(event) - len(AR)):
            V = compute_optical_flow(R_scaled.isel(time=slice(i, i + len(AR) + 1)).values, raft)
            nowcast_method = nowcasts.get_method("extrapolation")
            try:
                R_f_raft = nowcast_method(R[i + len(AR), :, :], V, timesteps=LEADTIME, measure_time=True)[0]
                R_f_raft = transformation.dB_transform(R_f_raft, threshold=-10.0, inverse=True)[0]
            except:
                R_f_raft = np.zeros((LEADTIME, data_dynamic.y.size, data_dynamic.x.size), dtype=float)
            
            event_raft.append(R_f_raft)

        ds_extrapolation = xr.Dataset(
            data_vars=dict(
                raft=(["forecast_reference_time", "leadtime", "y", "x"], np.asarray(event_raft)),
            ),
            coords=dict(
                forecast_reference_time=event[len(AR):],
                leadtime=LEADTIMES,
                x=(["x"], data_dynamic.x.data),
                y=(["y"], data_dynamic.y.data),
            ),
            attrs={},
        )

        ds_extrapolation.to_zarr(test_event_dir / f"raft_test_event_{idx}.zarr", mode="w")

    print("   ---> Elapsed time: {:.1f} hours ".format((time.time() - t_i) / 60 / 60))

    t_verification = time.time()

    list_ds_extrapolation = []
    for idx, event in enumerate(test_events):
        test_event_dir = benchmark_dir_path / f"test_event_{idx}"
        list_ds_extrapolation.append(xr.open_zarr(test_event_dir / f"raft_test_event_{idx}.zarr"))

    ds_extrapolation = xr.concat(list_ds_extrapolation, dim="forecast_reference_time")
    ds_extrapolation = xr_sel_coords_between(ds_extrapolation, **boundaries)
    ds_extrapolation = reshape_forecasts_for_verification(ds_extrapolation)

    # Verification in Switzerland
    boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
    data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr",
                                        timestep=5, boundaries=boundaries)
    for key in (ds_extrapolation.data_vars.keys()):
        print("Computing verification metrics for :", key)
        model_skills_dir = (benchmark_dir_path / "combined_ch" / "skills" / key)
        model_skills_dir.mkdir(parents=True, exist_ok=True)
        ds_temp = ds_extrapolation[[key]].rename({key: "feature"}) 
        verification_routine(ds_temp, data_dynamic, model_skills_dir, print_metrics=True)

    print("   ---> Elapsed time: {:.1f} hours ".format((time.time() - t_verification) / 60 / 60))

    print("========================================================================================")
    print(
        "- Benchmark generation and verification terminated. Elapsed time: {:.1f} hours ".format(
            (time.time() - t_i) / 60 / 60
        )
    )
        