import pathlib
import itertools
from typing import List, Tuple
import numpy as np
import pandas as pd
import xarray as xr

from pysteps.feature.blob import detection as blob_detection
from pysteps.feature.tstorm import detection as cell_detection
import matplotlib.pyplot as plt

zarr_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/zarr/")

ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")

ds_masked = ds.sel({"y": list(range(850, 450, -1)), "x": list(range(30, 320))})

def plot_patches(arr: np.ndarray, patches_idx: List[Tuple[int, int]]):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.imshow(arr, cmap='Greys_r')
    for idx in patches_idx:
        x, y = idx
        rect = plt.Rectangle((y, x), 128, 128, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.set_axis_off()
    plt.show()

def get_precipitation_pixels_in_patch(arr: np.ndarray, threshold: float, start_idx: Tuple[int, int]):
    pixels_idx = np.column_stack(np.where(arr > threshold))
    return [(start_idx[0]+idx[0], start_idx[1]+idx[1]) for idx in pixels_idx]


def process_patches(patches_idx: List[Tuple[int, int]], patches_elements: List[List[Tuple[int, int]]]):
    to_remove = []

    for comb in itertools.combinations(range(len(patches_idx)), 2):
        set1 = set(patches_elements[comb[0]])
        set2 = set(patches_elements[comb[1]])
        if len(set1.intersection(set2))/len(set1) > 0.90 and len(set1) > len(set2):
            to_remove.append(comb[1])
        elif len(set1.intersection(set2))/len(set2) > 0.90:
            to_remove.append(comb[0])


    patches_elements = [patch for i, patch in enumerate(patches_elements) if i not in to_remove]
    patches_idx = [patch for i, patch in enumerate(patches_idx) if i not in to_remove]

    return patches_idx, patches_elements       


def find_patches(arr: np.ndarray, threshold: float = 0.04, patch_size: int = 128, 
                 min_nb_elems_in_patch: int = 5, plot: bool = True):
    a = arr.copy()
    still_patches = True
    patches_idx = []
    patches_elements = []
    while still_patches:
        idx = np.unravel_index(np.argmax(a > threshold), a.shape)
        if a[idx] > threshold:
            print(a[idx])
            if idx[0] > (a.shape[0] - patch_size):
                idx = (a.shape[0] - patch_size, idx[1])
            if idx[1] > (a.shape[1] - patch_size):
                idx = (idx[0], a.shape[1] - patch_size)
            
            print([idx[0], idx[0]+patch_size, idx[1], idx[1]+patch_size])

            elems = get_precipitation_pixels_in_patch(arr[idx[0]:idx[0]+patch_size, 
                                                          idx[1]:idx[1]+patch_size], threshold, idx)
            
            if len(elems) > min_nb_elems_in_patch:
                patches_idx.append(idx)
                patches_elements.append(elems)
            a[idx[0]:idx[0]+patch_size, idx[1]:idx[1]+patch_size] = 0
        else:
            still_patches = False 
    if plot:
        plot_patches(arr > threshold, patches_idx)

    patches_idx, patches_elements = process_patches(patches_idx, patches_elements)

    if plot:
        plot_patches(arr > threshold, patches_idx)

    return patches_idx


def get_patch_stats(arr: np.ndarray, patches_idx: List[List[Tuple[int, int]]], patch_size: int = 128):
    stats = []

    for idx in patches_idx:
        patch = arr[idx[0]:idx[0]+patch_size, idx[1]:idx[1]+patch_size]
        stats.append({
            "Position": idx,
            "Max": np.max(patch),
            "Min": np.min(patch),
            "Mean": np.mean(patch)
        })
    
    return stats

selection = ds_masked.isel({"time": 3}).fillna(0)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.imshow((selection.precip.values > 0.04), cmap='Greys_r')
ax.set_axis_off()
plt.show()

patches_idx = find_patches(selection.precip.values, threshold=0.1)
stats = pd.DataFrame(get_patch_stats(selection.precip.values, patches_idx))
stats
