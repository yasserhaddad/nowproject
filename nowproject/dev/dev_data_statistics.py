import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import time
from nowproject.data.data_utils import prepare_data_dynamic
from dask.distributed import Client
from dask.diagnostics import ProgressBar

def get_unique_counts(arr: np.ndarray):
    unique, counts = np.unique(arr, return_counts=True)
    return np.array([dict(zip(unique, counts))], dtype="object")

if __name__ == '__main__':
    client = Client(n_workers=12)
    data_dir_path = Path("/ltenas3/0_Data/NowProject/")
    data_stats_dir_path = data_dir_path / "stats_5min"
    data_stats_dir_path.mkdir(exist_ok=True)

    boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
    data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr",
                                        boundaries=boundaries,
                                        timestep=5)
    data_dynamic = data_dynamic.where(data_dynamic < 200.0, np.nan)

    print("Computing value counts...")
    t_i = time.time()
    results = xr.apply_ufunc(get_unique_counts, 
                             data_dynamic.feature,
                             input_core_dims=[["x", "y"]],
                             output_core_dims=[["info"]], 
                             dask="parallelized",
                             vectorize=True,
                             output_dtypes=["object"],
                             dask_gufunc_kwargs={'output_sizes': {'info': 1}}
                            )
    with ProgressBar():
        a = results.compute()

    combined = {}
    for d in a.data.flatten():
        for i in d:
            if i in combined:
                combined[i] += d[i]
            else:
                combined[i] = d[i]
    
    counts = pd.DataFrame.from_dict(combined, orient="index").reset_index()
    counts.columns = ["value", "count"]
    counts.to_csv(data_stats_dir_path / "counts_values.csv", index=False)
    t_end = time.time()
    print("Elapsed time: {:.2f}h".format((t_end - t_i)/3600))

    print("Computing grid cell statistics...")
    t_i = time.time()
    mean_space = data_dynamic.mean(dim="time").rename({"feature": "mean"}).compute()
    max_space = data_dynamic.max(dim="time").rename({"feature": "max"}).compute()
    xr.merge([mean_space, max_space]).to_netcdf(data_stats_dir_path / "stats_space.nc")
    t_end = time.time()
    print("Elapsed time: {:.2f}h".format((t_end - t_i)/3600))


    print("Computing monthly grid cell statistics...")
    t_i = time.time()
    mean_space_month = data_dynamic.groupby("time.month").mean(dim="time")\
                                   .rename({"feature": "mean"}).compute()
    max_space_month = data_dynamic.groupby("time.month").max(dim="time")\
                                  .rename({"feature": "max"}).compute()
    xr.merge([mean_space_month, max_space_month]).to_netcdf(data_stats_dir_path / "stats_space_month.nc")
    t_end = time.time()
    print("Elapsed time: {:.2f}h".format((t_end - t_i)/3600))


    print("Computing yearly grid cell statistics...")
    t_i = time.time()
    mean_space_month = data_dynamic.groupby("time.year").mean(dim="time")\
                                   .rename({"feature": "mean"}).compute()
    max_space_month = data_dynamic.groupby("time.year").max(dim="time")\
                                  .rename({"feature": "max"}).compute()
    xr.merge([mean_space_month, max_space_month]).to_netcdf(data_stats_dir_path / "stats_space_year.nc")
    t_end = time.time()
    print("Elapsed time: {:.2f}h".format((t_end - t_i)/3600))









