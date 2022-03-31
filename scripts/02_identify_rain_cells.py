import pathlib
import xarray as xr

import skimage
from pysteps.feature.blob import detection
import matplotlib.pyplot as plt

zarr_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/zarr/")

ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")

ds_masked = ds.sel({"y": list(range(850, 450, -1)), "x": list(range(30, 320))})
selection = ds_masked.isel({"time": 0}).fillna(0)

blobs = detection(selection.precip, method="log", threshold=0.1, min_sigma=1, max_sigma=40, 
                  overlap=0.01, return_sigmas=True)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.imshow(selection.precip)
for blob in blobs:
    x, y, r = blob
    c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
    ax.add_patch(c)
ax.set_axis_off()
plt.show()