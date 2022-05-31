import xarray as xr
import numpy as np

from pathlib import Path
from pysteps import motion, nowcasts

from xverif import xverif
from nowproject.utils.config import (
    create_test_events_autoregressive_time_range
)
from nowproject.data.data_utils import prepare_data_dynamic
from xforecasting.predictions_autoregressive import reshape_forecasts_for_verification

from pysteps.utils import transformation

LEADTIME = 21
FREQ_IN_NS = 150000000000
LEADTIMES = np.arange(FREQ_IN_NS, FREQ_IN_NS*(LEADTIME + 1), FREQ_IN_NS, dtype="timedelta64[ns]")
AR = [-2, -1]

if __name__ == "__main__":
    data_dir_path = Path("/ltenas3/0_Data/NowProject/")
    test_events_path = Path("/home/haddad/nowproject/configs/events.json")

    boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
    data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr",
                                        boundaries=boundaries)
    test_events = create_test_events_autoregressive_time_range(test_events_path, len(AR))
    (data_dir_path / "benchmarks").mkdir(parents=True, exist_ok=True)

    steps_vel_pert_init = {
        "p_par": [2.31970635, 0.33734287, -2.64972861],
        "p_perp": [1.90769947, 0.33446594, -2.06603662]
    }

    for idx, event in enumerate(test_events):
        test_event_dir = data_dir_path / "benchmarks" / f"test_event_{idx}"
        test_event_dir.mkdir(exist_ok=True, parents=True)

        event_sprog = []
        event_steps = []
        oflow_method = motion.get_method("lucaskanade")
        R = data_dynamic.sel(time=event).feature.values
        R = transformation.dB_transform(R, threshold=0.1, zerovalue=-15.0)[0]
        R[~np.isfinite(R)] = -15.0
        V = oflow_method(R)
        for i in range(len(event) - len(AR)):
            nowcast_method = nowcasts.get_method("sprog")
            try:
                R_f_sprog = nowcast_method(R[i:i + len(AR) + 1, :, :], V, timesteps=LEADTIME, 
                                            R_thr=-10.0, num_workers=18, measure_time=True, 
                                            ar_order=len(AR))[0]
                R_f_sprog = transformation.dB_transform(R_f_sprog, threshold=-10.0, inverse=True)[0]
            except:
                R_f_sprog = np.zeros((LEADTIME, data_dynamic.y.size, data_dynamic.x.size), dtype=float)
            
            event_sprog.append(R_f_sprog)

            nowcast_method = nowcasts.get_method("steps")
            try:
                R_f_steps = nowcast_method(R[i:i + len(AR) + 1, :, :], V, timesteps=LEADTIME, 
                                        n_ens_members=24, n_cascade_levels=8, kmperpixel=1.0, ar_order=len(AR),
                                        R_thr=-10.0, timestep=2.5, num_workers=18, vel_pert_kwargs=steps_vel_pert_init,
                                        measure_time=True)
                if type(R_f_steps) == tuple:
                    R_f_steps = R_f_steps[0]

                R_f_steps = np.mean(R_f_steps, axis=0)
                R_f_steps = transformation.dB_transform(R_f_steps, threshold=-10.0, inverse=True)[0]
            except:
                R_f_steps = np.zeros((LEADTIME, data_dynamic.y.size, data_dynamic.x.size), dtype=float)

            event_steps.append(R_f_steps)

        ds_benchmark = xr.Dataset(
            data_vars=dict(
                sprog=(["forecast_reference_time", "leadtime", "y", "x"], np.asarray(event_sprog)),
                steps=(["forecast_reference_time", "leadtime", "y", "x"], np.asarray(event_steps)),
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

    for idx, event in enumerate(test_events):
        print("Test event", idx+1)
        test_event_dir = data_dir_path / "benchmarks" / f"test_event_{idx}"
        ds_benchmark = xr.open_zarr(test_event_dir / f"benchmark_test_event_{idx}.zarr")
        ds_benchmark = reshape_forecasts_for_verification(ds_benchmark)

        for key in (ds_benchmark.data_vars.keys()):
            skills_dir = (test_event_dir / "skills" / key)
            skills_dir.mkdir(exist_ok=True, parents=True)
            ds_temp = ds_benchmark[[key]].rename({key: "feature"})
            ds_det_cont_skill = xverif.deterministic(
                pred=ds_temp.chunk({"time": -1}),
                obs=data_dynamic.sel(time=ds_temp.time).chunk({"time": -1}),
                forecast_type="continuous",
                aggregating_dim="time",
            )
            # - Save deterministic continuous skills
            ds_det_cont_skill.to_netcdf((skills_dir/ "deterministic_continuous_spatial_skill.nc"))

            ds_det_cat_skill = xverif.deterministic(
                pred=ds_temp.chunk({"time": -1}),
                obs=data_dynamic.sel(time=ds_temp.time).chunk({"time": -1}),
                forecast_type="categorical",
                aggregating_dim="time",
                thr=0.1
            )
            # - Save deterministic categorical skills
            ds_det_cat_skill.to_netcdf((skills_dir / "deterministic_categorical_spatial_skill.nc"))
            
            ds_det_spatial_skill = xverif.deterministic(
                pred=ds_temp.chunk({"x": -1, "y": -1}),
                obs=data_dynamic.sel(time=ds_temp.time).chunk({"x": -1, "y": -1}),
                forecast_type="spatial",
                aggregating_dim=["x", "y"],
                thr=0.1,
                win_size=5
            )
            ds_det_spatial_skill.to_netcdf((skills_dir / "deterministic_spatial_skill.nc"))
            

            ds_cont_averaged_skill = ds_det_cont_skill.mean(dim=["y", "x"])
            ds_cat_averaged_skill = ds_det_cat_skill.mean(dim=["y", "x"])
            ds_spatial_average_skill = ds_det_spatial_skill.mean(dim=["time"])

            # - Save averaged skills
            ds_cont_averaged_skill.to_netcdf(skills_dir / "deterministic_continuous_global_skill.nc")
            ds_cat_averaged_skill.to_netcdf(skills_dir / "deterministic_categorical_global_skill.nc")
            ds_spatial_average_skill.to_netcdf(skills_dir / "deterministic_spatial_global_skill.nc")
            
            print("RMSE:")
            print(ds_cont_averaged_skill["feature"].sel(skill="RMSE").values)
            print("F1:")
            print(ds_cat_averaged_skill["feature"].sel(skill="F1").values)
            print("ACC:")
            print(ds_cat_averaged_skill["feature"].sel(skill="ACC").values)
            print("CSI:")
            print(ds_cat_averaged_skill["feature"].sel(skill="CSI").values)
            print("FSS:")
            print(ds_spatial_average_skill["feature"].sel(skill="F1").values)
