#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:46:53 2022

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
    xr_get_areas_labels,
    get_slice_size,
)
from nowproject.data.data_utils import prepare_data_dynamic

zarr_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/zarr/")
# ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")
boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
ds = prepare_data_dynamic(zarr_dir_path / "rzc_temporal_chunk.zarr",
                          boundaries=boundaries)
# ds_masked = ds.sel({"y": list(range(850, 450, -1)), "x": list(range(30, 320))})

# DEV 
data_array = ds['feature'].sel(time="2016-01-01T00:00:00")
# data_array = ds['precip'].isel(time=0)
 
# Plot
cmap = plt.get_cmap("Spectral").copy()
cmap.set_under(color="white")
p = data_array.plot.imshow(vmin=0.1, cmap=cmap)

# Get array
arr = data_array.data.copy()
intensity = arr 

# Labels settings
min_intensity_threshold = 0.1
max_intensity_threshold = 300
min_area_threshold = 36
max_area_threshold = np.Inf
 
from skimage.morphology import square
footprint_buffer = square(10)
footprint_buffer.shape
 
# Patch settings 
patch_size = (128,128)
centered_on = "centroid"
mask_value = 0
 
labels, n_labels, counts = get_areas_labels(arr,  
                                            min_intensity_threshold=min_intensity_threshold, 
                                            max_intensity_threshold=max_intensity_threshold, 
                                            min_area_threshold=min_area_threshold, 
                                            max_area_threshold=max_area_threshold, 
                                            footprint_buffer=footprint_buffer) 

da_labels, n_labels, counts = xr_get_areas_labels(data_array,  
                                                  min_intensity_threshold=min_intensity_threshold, 
                                                  max_intensity_threshold=max_intensity_threshold, 
                                                  min_area_threshold=min_area_threshold, 
                                                  max_area_threshold=max_area_threshold, 
                                                  footprint_buffer=footprint_buffer)  
 
labels = da_labels.data
    
da_labels.plot.imshow(cmap="Spectral", vmin=1, interpolation="none")
plt.show()

plt.imshow(labels, cmap=cmap, vmin=1, interpolation="none")
plt.colorbar()
plt.show()

labels
np.arange(1, n_labels+1)
counts
 
# %timeit labels, _,_ = get_areas_labels(arr=arr, min_intensity_threshold=1, min_area_threshold=10)                   # 122 ms
# %timeit labels, _,_ = get_areas_labels(arr=np.zeros((1000,1000)), min_intensity_threshold=1, min_area_threshold=10) # 24 ms

list_patch_slices, patch_statistics = get_patch_per_label(labels, 
                                                          intensity, 
                                                          patch_size = patch_size, 
                                                          # centered_on = "max",
                                                          # centered_on = "centroid",
                                                          centered_on = "center_of_mass",
                                                          patch_stats_fun = patch_stats_fun, 
                                                          mask_value = mask_value, 
                                                          )
# [get_slice_size(slc) for slc, _ in list_patch_slices]
# [get_slice_size(slc) for _, slc in list_patch_slices]
 
# Found upper left index
list_patch_upper_left_idx = [[slc.start for slc in list_slc] for list_slc in list_patch_slices]
list_patch_upper_left_idx

# Plot all bounding boxes 
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.imshow(intensity, cmap=cmap, vmin=0.1)
for y, x in list_patch_upper_left_idx:
    rect = plt.Rectangle((x, y), patch_size[0], patch_size[0], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
ax.set_axis_off()
plt.show()

#-----------------------------------------------------------------------------.
# Plot patches 
for i, (r_slice, c_slice) in enumerate(list_patch_slices):
    plt.imshow(intensity[r_slice, c_slice], cmap=cmap, vmin=0.1)
    plt.colorbar()
    plt.show() 

#-----------------------------------------------------------------------------.

# TODO: check for non-nan 
# TODO: check for non-overlapping 
# TODO: skip for nan, or search bbox with not nan is post-processing 
 
 