import os
import shutil
import time
from typing import Callable, List, Union
import torch
import zarr
import dask
import numpy as np
import xarray as xr
from nowproject.scalers import Scaler

from xforecasting.dataloader_autoregressive import (
    remove_unused_Y,
    get_aligned_ar_batch,
    AutoregressiveDataset,
    AutoregressiveDataLoader,
)

from xforecasting.utils.ar import (
    check_ar_settings,
    check_input_k,
    check_output_k,
)
from xforecasting.utils.io import (
    _get_feature_order,
    check_timesteps_format,
    check_no_duplicate_timesteps,
)
from xforecasting.utils.zarr import (
    check_rounding,
    write_zarr,
)
from xforecasting.utils.torch import (
    check_device,
    check_pin_memory,
    check_asyncronous_gpu_transfer,
    check_prefetch_in_gpu,
    check_prefetch_factor,
)

from xforecasting.predictions_autoregressive import (
    get_dict_Y_pred_selection,
    create_ds_forecast,
    rescale_forecasts,
    rescale_forecasts_and_write_zarr,
)

def AutoregressivePredictions(
    model,
    # Data
    data_dynamic: Union[xr.DataArray, xr.Dataset],
    data_static: Union[xr.DataArray, xr.Dataset] = None,
    data_bc: Union[xr.DataArray, xr.Dataset] = None,
    bc_generator: Callable = None,
    # AR_batching_function
    ar_batch_fun: Callable = get_aligned_ar_batch,
    # Scaler options
    scaler_transform: Scaler = None,
    scaler_inverse: Scaler = None,
    # Dataloader options
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    prefetch_in_gpu: bool = False,
    pin_memory: bool = False,
    asyncronous_gpu_transfer: bool = True,
    device: str = "cpu",
    # Autoregressive settings
    input_k: List[int] = [-3, -2, -1],
    output_k: List[int] = [0],
    forecast_cycle: int = 1,
    ar_iterations: int = 50,
    stack_most_recent_prediction: bool = True,
    # Prediction options
    forecast_reference_times: np.ndarray = None,
    keep_first_prediction: bool = True,
    ar_blocks: int = None,
    # Save options
    zarr_fpath: str = None,
    rounding: int = None,
    compressor: Union[str, dict] = "auto",
    chunks: Union[str, dict] = "auto",
):
    """Wrapper to generate weather forecasts following CDS Common Data Model (CDM).

    CDS coordinate             dtype               Synonims
    -------------------------------------------------------------------------
    - realization              int64
    - forecast_reference_time  datetime64[ns]      (base time)
    - leadtime                 timedelta64[ns]
    - lat                      float64
    - lon                      float64
    - time                     datetime64[ns]      (forecasted_time/valid_time)

    To convert to ECMWF Common Data Model use the following code:
    import cf2cdm
    cf2cdm.translate_coords(ds_forecasts, cf2cdm.ECMWF)

    Terminology
    - Forecasts reference time: The time of the analysis from which the forecast was made
    - (Validity) Time: The time represented by the forecast
    - Leadtime: The time interval between the forecast reference time and the (validity) time.

    Coordinates notes:
    - output_k = 0 correspond to the first forecast leadtime
    - leadtime = 0 is not forecasted. It correspond to the analysis forecast_reference_time
    - In the ECMWF CMD, forecast_reference_time is termed 'time', 'time' termed 'valid_time'!

    Prediction settings
    - ar_blocks = None (or ar_blocks = ar_iterations + 1) run all ar_iterations in a single run.
    - ar_blocks < ar_iterations + 1:  run ar_iterations per ar_block of ar_iteration
    """
    # Possible speed up: rescale only after all batch have been processed ...
    ##------------------------------------------------------------------------.
    with dask.config.set(scheduler="synchronous"):
        ## Checks arguments
        device = check_device(device)
        pin_memory = check_pin_memory(
            pin_memory=pin_memory, num_workers=num_workers, device=device
        )
        asyncronous_gpu_transfer = check_asyncronous_gpu_transfer(
            asyncronous_gpu_transfer=asyncronous_gpu_transfer, device=device
        )
        prefetch_in_gpu = check_prefetch_in_gpu(
            prefetch_in_gpu=prefetch_in_gpu, num_workers=num_workers, device=device
        )
        prefetch_factor = check_prefetch_factor(
            prefetch_factor=prefetch_factor, num_workers=num_workers
        )
        ##------------------------------------------------------------------------.
        # Check that autoregressive settings are valid
        # - input_k and output_k must be numpy arrays hereafter !
        input_k = check_input_k(input_k=input_k, ar_iterations=ar_iterations)
        output_k = check_output_k(output_k=output_k)
        check_ar_settings(
            input_k=input_k,
            output_k=output_k,
            forecast_cycle=forecast_cycle,
            ar_iterations=ar_iterations,
            stack_most_recent_prediction=stack_most_recent_prediction,
        )
        ar_iterations = int(ar_iterations)
        ##------------------------------------------------------------------------.
        ### Retrieve feature info of the forecast
        features = _get_feature_order(data_dynamic)

        ##------------------------------------------------------------------------.
        # Check Zarr settings
        WRITE_TO_ZARR = zarr_fpath is not None
        if WRITE_TO_ZARR:
            # - If zarr fpath provided, create the required folder
            if not os.path.exists(os.path.dirname(zarr_fpath)):
                os.makedirs(os.path.dirname(zarr_fpath))
            if os.path.exists(zarr_fpath):
                raise ValueError("An {} store already exists.")
            # - Set default chunks and compressors
            # ---> -1 to all optional dimensions (i..e nodes, lat, lon, ens, plevels,...)
            dims = list(data_dynamic.dims)
            dims_optional = np.array(dims)[
                np.isin(dims, ["time", "feature"], invert=True)
            ].tolist()
            default_chunks = {dim: -1 for dim in dims_optional}
            default_chunks["forecast_reference_time"] = 1
            default_chunks["leadtime"] = 1
            default_compressor = zarr.Blosc(cname="zstd", clevel=0, shuffle=2)
            # - Check rounding settings
            rounding = check_rounding(rounding=rounding, variable_names=features)
        ##------------------------------------------------------------------------.
        # Check ar_blocks
        if not isinstance(ar_blocks, (int, float, type(None))):
            raise TypeError("'ar_blocks' must be int or None.")
        if isinstance(ar_blocks, float):
            ar_blocks = int(ar_blocks)
        if not WRITE_TO_ZARR and isinstance(ar_blocks, int):
            raise ValueError("If 'zarr_fpath' not specified, 'ar_blocks' must be None.")
        if ar_blocks is None:
            ar_blocks = ar_iterations + 1
        if ar_blocks > ar_iterations + 1:
            raise ValueError("'ar_blocks' must be equal or smaller to 'ar_iterations'")
        PREDICT_AR_BLOCKS = ar_blocks != (ar_iterations + 1)

        ##------------------------------------------------------------------------.
        ### Define DataLoader subset_timesteps
        subset_timesteps = None
        if forecast_reference_times is not None:
            # Check forecast_reference_times
            forecast_reference_times = check_timesteps_format(forecast_reference_times)
            if len(forecast_reference_times) == 0:
                raise ValueError(
                    "If you don't want to specify specific 'forecast_reference_times', set it to None"
                )
            check_no_duplicate_timesteps(
                forecast_reference_times, var_name="forecast_reference_times"
            )
            # Ensure the temporal order of forecast_reference_times
            forecast_reference_times.sort()
            # Define subset_timesteps (aka idx_k=0 aka first forecasted timestep)
            t_res_timedelta = np.diff(data_dynamic.time.values)[0]
            subset_timesteps = (
                forecast_reference_times + -1 * max(input_k) * t_res_timedelta
            )
            # Redefine batch_size if larger than the number of forecast to generate
            # --> And set num_workers to 0 (only 1 batch to load ...)
            if batch_size >= len(forecast_reference_times):
                batch_size = len(forecast_reference_times)
                num_workers = 0

        ##------------------------------------------------------------------------.
        ### Create training Autoregressive Dataset and DataLoader
        dataset = AutoregressiveDataset(
            data_dynamic=data_dynamic,
            data_bc=data_bc,
            data_static=data_static,
            bc_generator=bc_generator,
            scaler=scaler_transform,
            # Dataset options
            subset_timesteps=subset_timesteps,
            training_mode=False,
            # Autoregressive settings
            input_k=input_k,
            output_k=output_k,
            forecast_cycle=forecast_cycle,
            ar_iterations=ar_iterations,
            stack_most_recent_prediction=stack_most_recent_prediction,
            # GPU settings
            device=device,
        )
        dataloader = AutoregressiveDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last_batch=False,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            prefetch_in_gpu=prefetch_in_gpu,
            pin_memory=pin_memory,
            asyncronous_gpu_transfer=asyncronous_gpu_transfer,
            device=device,
        )
        ##------------------------------------------------------------------------.
        # Retrieve custom ar_batch_fun fuction
        ar_batch_fun = dataset.ar_batch_fun

        assert features == dataset.feature_order["dynamic"]
        ### Start forecasting
        # - Initialize
        t_i = time.time()
        model.to(device)
        # - Set dropout and batch normalization layers to evaluation mode
        model.eval()
        list_ds = []
        FIRST_PREDICTION = True
        with torch.set_grad_enabled(False):
            ##--------------------------------------------------------------------.
            # Iterate along batches
            dataloader_iter = iter(dataloader)
            num_batches = len(dataloader_iter)
            batch_indices = range(num_batches)
            for batch_count in batch_indices:
                batch_dict = next(dataloader_iter)
                t_gen = time.time()
                ##----------------------------------------------------------------.
                ### Retrieve forecast informations
                dim_info_dynamic = batch_dict["dim_info"]["dynamic"]
                feature_order_dynamic = batch_dict["feature_order"]["dynamic"]
                forecast_time_info = batch_dict["forecast_time_info"]
                forecast_reference_times = forecast_time_info["forecast_reference_time"]
                dict_forecast_leadtime = forecast_time_info["dict_forecast_leadtime"]
                dict_forecast_rel_idx_Y = forecast_time_info["dict_forecast_rel_idx_Y"]
                leadtimes = np.unique(
                    np.stack(list(dict_forecast_leadtime.values())).flatten()
                )
                assert features == feature_order_dynamic
                ##----------------------------------------------------------------.
                ### Retrieve dictionary providing at each AR iteration
                #   the tensor slice indexing to obtain a "regular" forecasts
                if FIRST_PREDICTION:
                    dict_Y_pred_selection = get_dict_Y_pred_selection(
                        dim_info=dim_info_dynamic,
                        dict_forecast_rel_idx_Y=dict_forecast_rel_idx_Y,
                        keep_first_prediction=keep_first_prediction,
                    )
                    FIRST_PREDICTION = False
                ##----------------------------------------------------------------.
                ### Perform autoregressive forecasting
                dict_Y_predicted = {}
                dict_Y_predicted_per_leadtime = {}
                ar_counter_per_block = 0
                previous_block_ar_iteration = 0
                for ar_iteration in range(ar_iterations + 1):
                    # Retrieve X and Y for current AR iteration
                    # - Torch Y stays in CPU with training_mode=False
                    torch_X, _ = ar_batch_fun(
                        ar_iteration=ar_iteration,
                        batch_dict=batch_dict,
                        dict_Y_predicted=dict_Y_predicted,
                        device=device,
                        asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                    )

                    ##------------------------------------------------------------.
                    # Forward pass and store output for stacking into next AR iterations
                    dict_Y_predicted[ar_iteration] = model(torch_X, training=False)
                    ##------------------------------------------------------------.
                    # Select required tensor slices (along time dimension) for final forecast
                    if len(dict_Y_pred_selection[ar_iteration]) > 0:
                        for leadtime, subset_indexing in dict_Y_pred_selection[
                            ar_iteration
                        ]:
                            dict_Y_predicted_per_leadtime[leadtime] = (
                                dict_Y_predicted[ar_iteration][subset_indexing]
                                .cpu()
                                .numpy()
                            )
                    ##------------------------------------------------------------.
                    # Remove unnecessary variables on GPU
                    remove_unused_Y(
                        ar_iteration=ar_iteration,
                        dict_Y_predicted=dict_Y_predicted,
                        dict_Y_to_remove=batch_dict["dict_Y_to_remove"],
                    )
                    del torch_X
                    ##------------------------------------------------------------.
                    # The following code can be used to verify that no leak of memory occurs
                    # torch.cuda.synchronize()
                    # print("{}: {:.2f} MB".format(ar_iteration, torch.cuda.memory_allocated()/1000/1000))

                    ##------------------------------------------------------------.
                    # Create and save a forecast Dataset after each ar_block ar_iterations
                    ar_counter_per_block += 1
                    if ar_counter_per_block == ar_blocks:
                        block_slice = slice(
                            previous_block_ar_iteration, ar_iteration + 1
                        )
                        ds = create_ds_forecast(
                            dict_Y_predicted_per_leadtime=dict_Y_predicted_per_leadtime,
                            leadtimes=leadtimes[block_slice],
                            forecast_reference_times=forecast_reference_times,
                            data_dynamic=data_dynamic,
                            dim_info_dynamic=dim_info_dynamic,
                        )

                        # Reset ar_counter_per_block
                        ar_counter_per_block = 0
                        previous_block_ar_iteration = ar_iteration + 1
                        # --------------------------------------------------------.
                        # If predicting blocks of ar_iterations
                        # - Write AR blocks temporary to disk (and append progressively)
                        if PREDICT_AR_BLOCKS:  # (WRITE_TO_ZARR=True implicit)
                            tmp_ar_block_zarr_fpath = os.path.join(
                                os.path.dirname(zarr_fpath), "tmp_ar_blocks.zarr"
                            )
                            write_zarr(
                                zarr_fpath=tmp_ar_block_zarr_fpath,
                                ds=ds,
                                chunks=chunks,
                                default_chunks=default_chunks,
                                compressor=compressor,
                                default_compressor=default_compressor,
                                rounding=rounding,
                                consolidated=True,
                                append=True,
                                append_dim="leadtime",
                                show_progress=False,
                            )
                        # --------------------------------------------------------.
                ##--------------------------------------.-------------------------.
                # Clean memory
                del dict_Y_predicted
                del dict_Y_predicted_per_leadtime
                ##----------------------------------------------------------------.
                ### Post-processing
                t_post = time.time()
                # - Retransform data to original dimensions (and write to Zarr optionally)
                if WRITE_TO_ZARR:
                    if PREDICT_AR_BLOCKS:
                        # - Read the temporary ar_blocks saved on disk
                        ds = xr.open_zarr(tmp_ar_block_zarr_fpath)
                    if scaler_inverse is not None:
                        # TODO: Here an error occur if chunk forecast_reference_time > 1
                        # --> Applying the inverse scaler means processing each
                        #     forecast_reference_time separately
                        # ---> A solution would be to stack all forecasts together before
                        #      write to disk ... but this would consume memory and time.
                        rescale_forecasts_and_write_zarr(
                            ds=ds,
                            scaler=scaler_inverse,
                            zarr_fpath=zarr_fpath,
                            chunks=chunks,
                            default_chunks=default_chunks,
                            compressor=compressor,
                            default_compressor=default_compressor,
                            rounding=rounding,
                            consolidated=True,
                            append=True,
                            append_dim="forecast_reference_time",
                            show_progress=False,
                        )
                    else:
                        write_zarr(
                            zarr_fpath=zarr_fpath,
                            ds=ds,
                            chunks=chunks,
                            default_chunks=default_chunks,
                            compressor=compressor,
                            default_compressor=default_compressor,
                            rounding=rounding,
                            consolidated=True,
                            append=True,
                            append_dim="forecast_reference_time",
                            show_progress=False,
                        )
                    if PREDICT_AR_BLOCKS:
                        shutil.rmtree(tmp_ar_block_zarr_fpath)

                else:
                    if scaler_inverse is not None:
                        ds = rescale_forecasts(
                            ds=ds, scaler=scaler_inverse, reconcat=True
                        )
                    list_ds.append(ds)
                # -------------------------------------------------------------------.
                # Print prediction report
                tmp_time_gen = round(t_post - t_gen, 1)
                tmp_time_post = round(time.time() - t_post, 1)
                tmp_time_per_forecast = round(
                    (tmp_time_gen + tmp_time_post) / batch_size, 3
                )
                print(
                    " - Batch: {} / {} | Generation: {}s | Writing: {}s |"
                    "Single forecast computation: {}s ".format(
                        batch_count,
                        len(dataloader),
                        tmp_time_gen,
                        tmp_time_post,
                        tmp_time_per_forecast,
                    )
                )
            # ---------------------------------------------------------------------.
            # Remove the dataloader and dataset to avoid deadlocks
            del batch_dict
            del dataset
            del dataloader
            del dataloader_iter

    ##------------------------------------------------------------------------.
    # Re-read the forecast dataset
    if WRITE_TO_ZARR:
        ds_forecasts = xr.open_zarr(zarr_fpath, chunks="auto")
    else:
        ds_forecasts = xr.merge(list_ds)
    ##------------------------------------------------------------------------.
    print(
        "- Elapsed time for forecast generation: {:.2f} minutes".format(
            (time.time() - t_i) / 60
        )
    )
    ##------------------------------------------------------------------------.
    return ds_forecasts