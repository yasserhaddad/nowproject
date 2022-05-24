#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:42:32 2022

@author: ghiggi
"""
import time
import zarr
import pathlib
import itertools
from typing import List, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors 
from nowproject.dev.utils_patches import (
    get_areas_labels,
    get_patch_per_label,
    patch_stats_fun
)

from skimage.morphology import square
from nowproject.data.data_utils import prepare_data_dynamic
from nowproject.data.patches_utils import get_patch_info
 
# Set parallel settings 
# - LTESRV1: 24 max
# - LTESRV7: 32 max 
from dask.distributed import Client, progress, LocalCluster

if __name__ == '__main__':
    client = Client(n_workers=22) # process=False fails ! 

    data_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/")
    # ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")
    boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
    ds = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr",
                              boundaries=boundaries)

    # Labels settings
    min_intensity_threshold = 0.1
    max_intensity_threshold = 300
    min_area_threshold = 36
    max_area_threshold = np.Inf

    footprint_buffer = square(10)
    footprint_buffer.shape
    
    # Patch settings 
    patch_size = (128, 128)
    centered_on = "centroid"
    mask_value = 0
 
    # -----------------------------------------------------.
    # Compute for all patches    
    t_i = time.time()
    data_array = ds['feature']
    kwargs = {
            "min_intensity_threshold": min_intensity_threshold, 
            "max_intensity_threshold": max_intensity_threshold, 
            "min_area_threshold": min_area_threshold, 
            "max_area_threshold": max_area_threshold, 
            "footprint_buffer": footprint_buffer,
            "patch_size": patch_size, 
            "centered_on": centered_on,
            "patch_stats_fun": patch_stats_fun,
            }
    results = xr.apply_ufunc(get_patch_info, 
                            data_array, 
                            input_core_dims=[["y", "x"]],
                            output_core_dims=[["info"]],
                            kwargs=kwargs, 
                            dask="parallelized",
                            vectorize=True,
                            output_dtypes=["object"],
                            dask_gufunc_kwargs={'output_sizes': {'info': 1}}
                            )

    a  = results.compute()   

    list_df = []
    for i in range(len(a.data)):
        df_json_str = a.data[i][0]
        timestep = a['time'].data[i]
        if df_json_str != '{}':
            df = pd.read_json(df_json_str)
            df["time"] = timestep
            list_df.append(df)

    df_all = pd.concat(list_df, ignore_index=True)
    df_all.to_parquet(data_dir_path / "rzc_cropped_patches.parquet")

    # df_indices = df_all.groupby("time")["upper_left_idx"].apply(lambda x: ', '.join(x))

    t_end = time.time()
    print("Elapsed time: {:.2f}h".format((t_end - t_i)/3600))