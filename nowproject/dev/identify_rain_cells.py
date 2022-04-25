import time
import zarr
import pathlib
import itertools
from typing import List, Tuple
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from xforecasting.utils.zarr import write_zarr

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


def extract_patches(arr: np.ndarray, threshold: float = 0.04, patch_size: int = 128, 
                 min_nb_elems_in_patch: int = 5, plot: bool = True):
    a = arr.copy()
    still_patches = True
    patches_idx = []
    patches_elements = []
    while still_patches:
        idx = np.unravel_index(np.argmax(a > threshold), a.shape)
        if a[idx] > threshold:
            if idx[0] > (a.shape[0] - patch_size):
                idx = (a.shape[0] - patch_size, idx[1])
            if idx[1] > (a.shape[1] - patch_size):
                idx = (idx[0], a.shape[1] - patch_size)

            elems = get_precipitation_pixels_in_patch(arr[idx[0]:idx[0]+patch_size, idx[1]:idx[1]+patch_size], 
                                                      threshold, idx)
            contains_nan = np.isnan(np.sum(arr[idx[0]:idx[0]+patch_size, idx[1]:idx[1]+patch_size]))

            if len(elems) > min_nb_elems_in_patch and not contains_nan:
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

    result = np.asarray([",".join([f"{idx[0]}-{idx[1]}" for idx in patches_idx])]) \
                if len(patches_idx) > 0 else np.asarray([""])

    return result


def convert_str_patch_idx_to_int(patches_idx):
    return [(int(idx[0]), int(idx[1])) for idx in [i.split("-")for i in patches_idx.split(",")]]

def get_patch_stats(arr: np.ndarray, patch_idx: List[int], patch_size: int = 128):
    patch = arr[patch_idx[0]:patch_idx[0]+patch_size, patch_idx[1]:patch_idx[1]+patch_size]
    return {
        "Position": patch_idx,
        "Max": np.max(patch),
        "Min": np.min(patch),
        "Mean": np.mean(patch),
        "Area >= 1": len(np.column_stack(np.where(patch >= 1))),
        "Area >= 5": len(np.column_stack(np.where(patch >= 5))),
        "Area >= 20": len(np.column_stack(np.where(patch >= 20))),
        "Sum": np.sum(patch),
        "Dry-Wet Area Ratio": len(np.column_stack(np.where(patch > 0.04)))/(patch_size*patch_size),
    }


def parallel_extraction(data_array: xr.DataArray, 
                        threshold: float = 0.04, 
                        patch_size: int = 128, 
                        min_nb_elems_in_patch: int = 5, 
                        plot: bool = False):
    
    kwargs = {
        "threshold": threshold, 
        "patch_size": patch_size, 
        "min_nb_elems_in_patch": min_nb_elems_in_patch, 
        "plot": plot
    }

    return xr.apply_ufunc(extract_patches, 
                          data_array, 
                          input_core_dims=[["y", "x"]],
                          output_core_dims=[["indices"]],
                          kwargs=kwargs, 
                          dask="parallelized",
                          vectorize=True,
                          output_dtypes=["object"],
                          dask_gufunc_kwargs={'output_sizes': {'indices': 1}}
                        )

if __name__ == "__main__":
    zarr_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/zarr/")

    ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")

    ds_masked = ds.sel({"y": list(range(850, 450, -1)), "x": list(range(30, 320))})

    # selection = ds_masked.isel({"time": 3})

    # fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # ax.imshow((selection.precip.values > 0.04), cmap='Greys_r')
    # ax.set_axis_off()
    # plt.show()

    # patches_idx = extract_patches(selection.precip.values, threshold=0.1)
    # # stats = pd.DataFrame(get_patch_stats(selection.precip.values, patches_idx))
    # if patches_idx[0] != "":
    #     stats = pd.DataFrame([get_patch_stats(selection.precip.values, idx) for idx in convert_str_patch_idx_to_int(patches_idx)])
    # else:
    #     stats = pd.DataFrame()
    # stats

    start = time.time()
    patches = parallel_extraction(data_array=ds_masked.precip).compute()
    print("{:.2f}".format(time.time() - start))
    patches = patches.astype(str).squeeze()
    patches.attrs = {"patch_size": 128}
    
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    
    write_zarr(
        (zarr_dir_path / "rzc_patches_temporal_chunk.zarr").as_posix(),
        patches,
        chunks={"time": 25},
        compressor=compressor,
        rounding=None,
        encoding=None,
        consolidated=True,
        append=False,
        show_progress=True,
    )