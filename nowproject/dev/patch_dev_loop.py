#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:01:32 2022

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
    patch_stats_fun,
    get_slice_size
)

zarr_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/zarr/")
ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")

# Labels settings
min_intensity_threshold = 0.1
max_intensity_threshold = 300
min_area_threshold = 36
max_area_threshold = np.Inf
# footprint_buffer = None 

from skimage.morphology import square
footprint_buffer = square(10)
footprint_buffer.shape
 
# Patch settings 
patch_size = (128, 128)
centered_on = "centroid"
mask_value = 0

#-----------------------------------------------------------------------------.
# Loop over
l_time = []
for i in range(1000):
    print("iter:", i)
    t_i = time.time()

    # Get array
    data_array = ds['precip'].isel(time=i).compute()
    arr = data_array.data.copy()
    intensity = arr 
 
    # Label area 
    labels, n_labels, counts = get_areas_labels(arr,  
                                                min_intensity_threshold=min_intensity_threshold, 
                                                max_intensity_threshold=max_intensity_threshold, 
                                                min_area_threshold=min_area_threshold, 
                                                max_area_threshold=max_area_threshold, 
                                                footprint_buffer=footprint_buffer) 
    
    if n_labels > 0:
        # Get patches 
        list_patch_slices, patch_statistics = get_patch_per_label(labels, 
                                                                  intensity, 
                                                                  patch_size=patch_size, 
                                                                  centered_on=centered_on,
                                                                  patch_stats_fun=patch_stats_fun, 
                                                                  mask_value=mask_value,
                                                                  )
        # Found upper left index
        list_patch_upper_left_idx = [[slc.start for slc in list_slc] for list_slc in list_patch_slices]
        list_patch_upper_left_idx
        
        assert not np.any(np.array([get_slice_size(slc) for slc, _ in list_patch_slices]) != 128)
        assert not np.any(np.array([get_slice_size(slc) for _, slc in list_patch_slices]) != 128)  
        
        # Time execution 
        t_f = time.time()
        t_elapsed = t_f-t_i
        print(t_elapsed)
        l_time.append(t_elapsed)
                
        #----------------------------------------------------------------------.
        # Plot all bounding boxes 
        # cmap = plt.get_cmap("Spectral").copy()
        # cmap.set_under(color="white")
        # fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        # ax.imshow(intensity, cmap=cmap, vmin=0.1)
        # for y, x in list_patch_upper_left_idx:
        #     rect = plt.Rectangle((x, y), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # ax.set_axis_off()
        # plt.show()
            
        #----------------------------------------------------------------------.


n_timesteps = 1262592
expected_s = 562/1000*n_timesteps
expected_time = expected_s/60/60/24 

#-----------------------------------------------------------------------------.

pd.DataFrame(patch_statistics)  

upper_left_str = [str(row) + "-" + str(col) for row, col in list_patch_upper_left_idx]

from nowproject.utils_patches import get_upper_left_idx_from_str
get_upper_left_idx_from_str(upper_left_str)
 
# patch_statistics = [patch_stats_fun(intensity[r_slice, c_slice]) for r_slice, c_slice in list_patch_slices]

# intensity
# r_slice, c_slice = list_patch_slices[0]
# patch = intensity[r_slice, c_slice]


 

 