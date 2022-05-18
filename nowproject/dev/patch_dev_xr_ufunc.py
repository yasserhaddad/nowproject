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
 
# Set parallel settings 
# - LTESRV1: 24 max
# - LTESRV7: 32 max 
from dask.distributed import Client, progress
client = Client(n_workers=22) # process=False fails ! 

zarr_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/zarr/")
ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")

# Labels settings
min_intensity_threshold = 0.1
max_intensity_threshold = 300
min_area_threshold = 36
max_area_threshold = np.Inf

from skimage.morphology import square
footprint_buffer = square(10)
footprint_buffer.shape
 
# Patch settings 
patch_size = (128, 128)
centered_on = "centroid"
mask_value = 0

#-----------------------------------------------------------------------------.
# Define patch info extraction 
def get_patch_info(arr,
                   min_intensity_threshold=-np.inf, 
                   max_intensity_threshold= np.inf, 
                   min_area_threshold=1, 
                   max_area_threshold=np.inf,
                   footprint_buffer=None,
                   patch_size: tuple = (128, 128), 
                   centered_on = "center_of_mass",  
                   mask_value = 0, 
                   patch_stats_fun = None, 
                  ):
     # Label area 
     labels, n_labels, counts = get_areas_labels(arr,  
                                                 min_intensity_threshold=min_intensity_threshold, 
                                                 max_intensity_threshold=max_intensity_threshold, 
                                                 min_area_threshold=min_area_threshold, 
                                                 max_area_threshold=max_area_threshold, 
                                                 footprint_buffer=footprint_buffer) 
     if n_labels > 0:
         # Get patches 
         list_patch_slices, patch_statistics = get_patch_per_label(labels=labels, 
                                                                   intensity=arr,
                                                                   patch_size=patch_size, 
                                                                   centered_on=centered_on,
                                                                   patch_stats_fun=patch_stats_fun, 
                                                                  )
         # Found upper left index
         list_patch_upper_left_idx = [[slc.start for slc in list_slc] for list_slc in list_patch_slices]
         upper_left_str = [str(row) + "-" + str(col) for row, col in list_patch_upper_left_idx]
         
         # Define data.frame 
         df = pd.DataFrame(patch_statistics) 
         df['upper_left_idx'] = upper_left_str
     else: 
         df = pd.DataFrame()
     
     # Hack to return df
     out_str = df.to_json()
     return np.array([out_str], dtype="object")
 
# -----------------------------------------------------.
# Compute for all patches 
                      
t_i = time.time()
data_array = ds['precip'].isel(time=slice(0,1000))
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
    timestep = a['time'].data[0]
    if df_json_str != '{}':
        df = pd.read_json(df_json_str)
        df["time"] = timestep
        list_df.append(df)

df_all = pd.concat(list_df, ignore_index=True)

# np.unique(df_all['Patch Area'])
# np.unique(df_all['Patch Area'], return_counts=True)
 
t_f = time.time()   
t_elapsed = t_f - t_i 
print("Total:", t_elapsed)  

# 59 sec with processes=True
# xxx sec with processes=False

np.unique(df_all[df_all['Patch Area'] == 16128]['time'])
 
