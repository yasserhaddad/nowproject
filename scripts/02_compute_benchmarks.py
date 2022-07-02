import time
import xarray as xr
import numpy as np

from pathlib import Path
from pysteps import motion, nowcasts

from nowproject.config import (
    create_test_events_autoregressive_time_range
)
from nowproject.data.data_utils import prepare_data_dynamic, xr_sel_coords_between
from xforecasting.predictions_autoregressive import reshape_forecasts_for_verification

from pysteps.utils import transformation
from nowproject.verification.verification import verification_routine


NUM_WORKERS = 15
LEADTIME = 12
FREQ_IN_NS = 150000000000 * 2
LEADTIMES = np.arange(FREQ_IN_NS, FREQ_IN_NS*(LEADTIME + 1), FREQ_IN_NS, dtype="timedelta64[ns]")
AR = [-2, -1]


if __name__ == "__main__":
    t_i = time.time()

    data_dir_path = Path("/ltenas3/0_Data/NowProject/")
    test_events_path = Path("/home/haddad/nowproject/configs/subset_test_events.json")

    boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
    timestep = 5
    data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr",
                                        timestep=timestep)
    test_events = create_test_events_autoregressive_time_range(test_events_path, len(AR), 
                                                               freq="5min")
    benchmark_dir_path = data_dir_path / "benchmarks" / "5min_full"
    benchmark_dir_path.mkdir(parents=True, exist_ok=True)

    steps_vel_pert_init = {
        "p_par": [2.31970635, 0.33734287, -2.64972861],
        "p_perp": [1.90769947, 0.33446594, -2.06603662]
    }

    for idx, event in enumerate(test_events):
        test_event_dir = benchmark_dir_path / f"test_event_{idx}"
        test_event_dir.mkdir(exist_ok=True, parents=True)

        event_sprog = []
        event_steps_mean = []
        event_steps_median = []

        oflow_method = motion.get_method("lucaskanade")
        R = data_dynamic.sel(time=event).feature.values
        R = transformation.dB_transform(R, threshold=0.1, zerovalue=-15.0)[0]
        R[~np.isfinite(R)] = -15.0
        V = oflow_method(R)

        for i in range(len(event) - len(AR)):
            nowcast_method = nowcasts.get_method("sprog")
            try:
                R_f_sprog = nowcast_method(R[i:i + len(AR) + 1, :, :], V, timesteps=LEADTIME, 
                                            R_thr=-10.0, num_workers=NUM_WORKERS, measure_time=True, 
                                            ar_order=len(AR))[0]
                R_f_sprog = transformation.dB_transform(R_f_sprog, threshold=-10.0, inverse=True)[0]
            except:
                R_f_sprog = np.zeros((LEADTIME, data_dynamic.y.size, data_dynamic.x.size), dtype=float)
            
            event_sprog.append(R_f_sprog)

            nowcast_method = nowcasts.get_method("steps")
            try:
                R_f_steps = nowcast_method(R[i:i + len(AR) + 1, :, :], V, timesteps=LEADTIME, 
                                        n_ens_members=24, n_cascade_levels=8, kmperpixel=1.0, ar_order=len(AR),
                                        R_thr=-10.0, timestep=timestep, num_workers=NUM_WORKERS, vel_pert_kwargs=steps_vel_pert_init,
                                        measure_time=True)
                if type(R_f_steps) == tuple:
                    R_f_steps = R_f_steps[0]

                R_f_steps_median = np.median(R_f_steps, axis=0)
                R_f_steps_median = transformation.dB_transform(R_f_steps_median, 
                                                               threshold=-10.0, inverse=True)[0]

                R_f_steps_mean = np.mean(R_f_steps, axis=0)
                R_f_steps_mean = transformation.dB_transform(R_f_steps_mean, threshold=-10.0, inverse=True)[0]
            except:
                R_f_steps_mean = np.zeros((LEADTIME, data_dynamic.y.size, data_dynamic.x.size), dtype=float)
                R_f_steps_median = np.zeros((LEADTIME, data_dynamic.y.size, data_dynamic.x.size), dtype=float)

            event_steps_mean.append(R_f_steps_mean)
            event_steps_median.append(R_f_steps_median)

        ds_benchmark = xr.Dataset(
            data_vars=dict(
                sprog=(["forecast_reference_time", "leadtime", "y", "x"], np.asarray(event_sprog)),
                steps_mean=(["forecast_reference_time", "leadtime", "y", "x"], np.asarray(event_steps_mean)),
                steps_median=(["forecast_reference_time", "leadtime", "y", "x"], np.asarray(event_steps_median)),
            ),
            coords=dict(
                forecast_reference_time=event[len(AR):],
                leadtime=LEADTIMES,
                x=(["x"], data_dynamic.x.data),
                y=(["y"], data_dynamic.y.data),
            ),
            attrs={},
        )

        ds_benchmark.to_zarr(test_event_dir / f"benchmark_test_event_{idx}.zarr", mode="w")

    print("   ---> Elapsed time: {:.1f} hours ".format((time.time() - t_i) / 60 / 60))

    t_verification = time.time()

    list_ds_benchmark = []
    for idx, event in enumerate(test_events):
        test_event_dir = benchmark_dir_path / f"test_event_{idx}"
        list_ds_benchmark.append(xr.open_zarr(test_event_dir / f"benchmark_test_event_{idx}.zarr"))

    # Verification on entire domain
    ds_benchmark = xr.concat(list_ds_benchmark, dim="forecast_reference_time")
    ds_benchmark = reshape_forecasts_for_verification(ds_benchmark)
    for key in (ds_benchmark.data_vars.keys()):
        print("Compute verification metrics for :", key)
        model_skills_dir = (benchmark_dir_path / "combined" / "skills" / key)
        ds_temp = ds_benchmark[[key]].rename({key: "feature"}) 
        verification_routine(ds_temp, data_dynamic, model_skills_dir, print_metrics=True)

    ds_benchmark = xr.concat(list_ds_benchmark, dim="forecast_reference_time")
    ds_benchmark = xr_sel_coords_between(ds_benchmark, **boundaries)
    ds_benchmark = reshape_forecasts_for_verification(ds_benchmark)

    # Verification in Switzerland
    boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
    data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr",
                                        timestep=5, boundaries=boundaries)
    for key in (ds_benchmark.data_vars.keys()):
        print("Computing verification metrics for :", key)
        model_skills_dir = (benchmark_dir_path / "combined_ch" / "skills" / key)
        ds_temp = ds_benchmark[[key]].rename({key: "feature"}) 
        verification_routine(ds_temp, data_dynamic, model_skills_dir, print_metrics=True)

    print("   ---> Elapsed time: {:.1f} hours ".format((time.time() - t_verification) / 60 / 60))

    print("========================================================================================")
    print(
        "- Benchmark generation and verification terminated. Elapsed time: {:.1f} hours ".format(
            (time.time() - t_i) / 60 / 60
        )
    )
        