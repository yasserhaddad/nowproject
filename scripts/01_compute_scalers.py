import os
import gc
from pathlib import Path
import time
import dask
from xscaler import GlobalMinMaxScaler, GlobalStandardScaler, AnomalyScaler, Climatology
import xarray as xr

def compute_scalers(data_dir_path, client, DataArrayFormat=False, force=True):
    dynamic_fpath = data_dir_path / "zarr" / "rzc_temporal_chunk.zarr"

    if DataArrayFormat:
        dask_chunks = {
            "feature": 1,
            "time": 576, 
            "x": -1, 
            "y": -1
        }
    else:
        dask_chunks = {
            "time": 576, 
            "x": -1, 
            "y": -1
        }
    
    variable_dim = "feature" if DataArrayFormat else None

    print("- Reading Datasets")
    data_dynamic = xr.open_zarr(dynamic_fpath, chunks = dask_chunks)
    if DataArrayFormat:
        data_dynamic = data_dynamic["data"] 
    
    ##------------------------------------------------------------------------.
    #### Define Global Standard Scaler
    print("- Compute Global Standard Scaler")
    dynamic_scaler = GlobalStandardScaler(data=data_dynamic, variable_dim=variable_dim)
    dynamic_scaler.fit()
    dynamic_scaler.save((data_dir_path / "Scalers" / "GlobalStandardScaler_dynamic.nc").as_posix(), force=force)
    del dynamic_scaler
    client.run(gc.collect)

    ##------------------------------------------------------------------------.
    #### Define Global MinMax Scaler
    print("- Compute  Global MinMax Scaler")
    dynamic_scaler = GlobalMinMaxScaler(data=data_dynamic, variable_dim=variable_dim)
    dynamic_scaler.fit()
    dynamic_scaler.save((data_dir_path / "Scalers" / "GlobalMinMaxScaler_dynamic.nc").as_posix(), force=force)
    del dynamic_scaler
    client.run(gc.collect)


if __name__ == '__main__':
    ##------------------------------------------------------------------------.
    ### Set dask configs 
    # - By default, Xarray and dask.array use thee multi-threaded scheduler (dask.config.set(scheduler='threads')
    # - 'num_workers' defaults to the number of cores
    # - dask.config.set(scheduler='threads') # Uses a ThreadPoolExecutor in the local process
    # - dask.config.set(scheduler='processes') # Uses a ProcessPoolExecutor to spread work between processes
    from dask.distributed import Client
    client = Client(processes=False)     
    # - Set array.chunk-size default
    dask.config.set({"array.chunk-size": "1024 MiB"})
    # - Avoid to split large dask chunks 
    dask.config.set(**{'array.slicing.split_large_chunks': False})

    DataArrayFormat = False

    data_dir_path = "/ltenas3/0_Data/NowProject/"
    print("==================================================================")
    print("- Computing scalers")
    t_i = time.time()
    #----------------------------------------------------------------------.
    compute_scalers(Path(data_dir_path), client, DataArrayFormat=DataArrayFormat)
    #----------------------------------------------------------------------.
    # Report elapsed time 
    print("---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i)/60))
    print("==================================================================")