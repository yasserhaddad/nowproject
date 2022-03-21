import pathlib
import xarray as xr
from xforecasting.dataloader_autoregressive import AutoregressiveDataset, AutoregressiveDataLoader, autoregressive_collate_fn

zarr_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/zarr_test/")

ds = xr.open_zarr(zarr_dir_path / "rzc_temporal_chunk.zarr")
ds = ds.rename({"precip": "feature"})

dataset = AutoregressiveDataset(ds, 
                                input_k=[-3, -2, -1],
                                output_k=[0],
                                forecast_cycle=1,
                                ar_iterations=6,
                                stack_most_recent_prediction=True)

dataloader = AutoregressiveDataLoader(
    dataset,
    batch_size=64,
    drop_last_batch=True,
    shuffle=True,
    shuffle_seed=69,
    num_workers=0,
    pin_memory=False,
    prefetch_in_gpu=False,
    prefetch_factor=2,
    asyncronous_gpu_transfer=True,
    device="cpu",
    verbose=False,
)

