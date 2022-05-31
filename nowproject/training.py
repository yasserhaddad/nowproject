#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:15:23 2022

@author: ghiggi
"""
import os
import torch
import time
import pickle
import dask
from xforecasting.dataloader_autoregressive import (
    AutoregressiveDataset,
    AutoregressiveDataLoader,
    get_aligned_ar_batch,
    remove_unused_Y,
    cylic_iterator,
)
from xforecasting.utils.ar import (
    check_ar_settings,
    check_input_k,
    check_output_k,
)
from xforecasting.utils.torch import (
    check_device,
    check_pin_memory,
    check_asyncronous_gpu_transfer,
    check_prefetch_in_gpu,
    check_prefetch_factor,
    check_ar_training_strategy,
    get_time_function,
)
from xforecasting.training_info import AR_TrainingInfo
from xforecasting.utils.xr import xr_is_aligned
from xforecasting.utils.swag import bn_update_with_loader
from xforecasting.training_autoregressive import timing_AR_Training

from nowproject.dataloader import AutoregressivePatchLearningDataset, AutoregressivePatchLearningDataLoader

def AutoregressiveTraining(
    model,
    model_fpath,
    # Loss settings
    criterion,
    reshape_tensors_4_loss,
    channels_first,
    ar_scheduler,
    early_stopping,
    optimizer,
    # Data
    training_data_dynamic,
    training_data_patches=None,
    training_data_bc=None,
    data_static=None,
    validation_data_dynamic=None,
    validation_data_patches=None,
    validation_data_bc=None,
    bc_generator=None,
    scaler=None,
    # AR_batching_function
    ar_batch_fun=get_aligned_ar_batch,
    # Dataloader options
    prefetch_in_gpu=False,
    prefetch_factor=2,
    drop_last_batch=True,
    shuffle=True,
    shuffle_seed=69,
    num_workers=0,
    pin_memory=False,
    asyncronous_gpu_transfer=True,
    # Autoregressive settings
    input_k=[-3, -2, -1],
    output_k=[0],
    forecast_cycle=1,
    ar_iterations=6,
    stack_most_recent_prediction=True,
    # Training settings
    ar_training_strategy="AR",
    lr_scheduler=None,
    training_batch_size=128,
    validation_batch_size=128,
    epochs=10,
    scoring_interval=10,
    save_model_each_epoch=False,
    ar_training_info=None,
    # SWAG settings
    swag=False,
    swag_model=None,
    swag_freq=10,
    swa_start=8,
    # GPU settings
    device="cpu",
):
    """AutoregressiveTraining.

    ar_batch_fun : callable
            Custom function that batch/stack together data across AR iterations.
            The custom function must return a tuple of length 2 (X, Y), but X and Y
            can be whatever desired objects (torch.Tensor, dict of Tensor, ...).
            The custom function must have the following arguments:
                def ar_batch_fun(ar_iteration, batch_dict, dict_Y_predicted,
                                 device = 'cpu', asyncronous_gpu_transfer = True)
            The default ar_batch_fun function is the pre-implemented get_aligned_ar_batch() which return
            two torch.Tensor: one for X (input) and one four Y (output). Such function expects
            the dynamic and bc batch data to have same dimensions and shape.
    if early_stopping=None, no ar_iteration update
    """
    with dask.config.set(scheduler="synchronous"):
        ##--------------------------------------------------------------------.
        time_start_training = time.time()
        separator = 88
        ## Checks arguments
        device = check_device(device)
        pin_memory = check_pin_memory(pin_memory=pin_memory,
                                      num_workers=num_workers,
                                      device=device)

        asyncronous_gpu_transfer = check_asyncronous_gpu_transfer(asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                                                                  device=device)
        prefetch_in_gpu = check_prefetch_in_gpu(prefetch_in_gpu=prefetch_in_gpu,
                                                num_workers=num_workers,
                                                device=device)
        prefetch_factor = check_prefetch_factor(prefetch_factor=prefetch_factor,
                                                num_workers=num_workers)
        ar_training_strategy = check_ar_training_strategy(ar_training_strategy)
        ##--------------------------------------------------------------------.
        # Check ar_scheduler
        if len(ar_scheduler.ar_weights) > ar_iterations + 1:
            n_weights = len(ar_scheduler.ar_weights)
            raise ValueError(f"The AR scheduler has {n_weights} AR weights, but ar_iterations is specified to be {ar_iterations}")
        if ar_iterations == 0:
            if ar_scheduler.method != "constant":
                print("Since 'ar_iterations' is 0, ar_scheduler 'method' is changed to 'constant'.")
                ar_scheduler.method = "constant"
        ##--------------------------------------------------------------------.
        # Check that autoregressive settings are valid
        # - input_k and output_k must be numpy arrays hereafter !
        print("- Defining AR settings:")
        input_k = check_input_k(input_k=input_k, ar_iterations=ar_iterations)
        output_k = check_output_k(output_k=output_k)
        check_ar_settings(
            input_k=input_k,
            output_k=output_k,
            forecast_cycle=forecast_cycle,
            ar_iterations=ar_iterations,
            stack_most_recent_prediction=stack_most_recent_prediction,
        )
        ##--------------------------------------------------------------------.
        # Check training data
        if training_data_dynamic is None:
            raise ValueError("'training_data_dynamic' must be provided !")

        ##--------------------------------------------------------------------.
        ## Check validation data
        if validation_data_dynamic is not None:
            if not xr_is_aligned(training_data_dynamic, validation_data_dynamic, exclude="time"):
                raise ValueError("training_data_dynamic' and 'validation_data_dynamic' does not"
                                 "share same dimensions (order and values)(excluding 'time').")
        if validation_data_bc is not None:
            if training_data_dynamic is None:
                raise ValueError("If 'validation_data_bc' is provided, also 'training_data_dynamic' must be specified.")
            if not xr_is_aligned(training_data_bc, validation_data_bc, exclude="time"):
                raise ValueError("training_data_bc' and 'validation_data_bc' does not"
                                 "share same dimensions (order and values)(excluding 'time').")

        ##--------------------------------------------------------------------.
        ## Check early stopping
        if validation_data_dynamic is None:
            if early_stopping is not None:
                if early_stopping.stopping_metric == "total_validation_loss":
                    print("Validation dataset is not provided. "
                          "Stopping metric of early_stopping set to 'total_training_loss'")
                    early_stopping.stopping_metric = "total_training_loss"
        ##--------------------------------------------------------------------.
        # Ensure criterion and model are on device
        model.to(device)
        criterion.to(device)
        ##--------------------------------------------------------------------.
        # Zeros gradients
        optimizer.zero_grad(set_to_none=True)
        ##--------------------------------------------------------------------.
        ### Create Datasets
        t_i = time.time()
        training_ds = AutoregressivePatchLearningDataset(
            data_dynamic=training_data_dynamic,
            data_bc=training_data_bc,
            data_patches=training_data_patches,
            data_static=data_static,
            bc_generator=bc_generator,
            scaler=scaler,
            # Custom AR batching function
            ar_batch_fun=ar_batch_fun,
            training_mode=True,
            # Autoregressive settings
            input_k=input_k,
            output_k=output_k,
            forecast_cycle=forecast_cycle,
            ar_iterations=ar_scheduler.current_ar_iterations,
            stack_most_recent_prediction=stack_most_recent_prediction,
            # Timesteps
            subset_timesteps=(training_data_patches.time.values if training_data_patches is not None else None),
            # GPU settings
            device=device,
        )
        if validation_data_dynamic is not None:
            validation_ds = AutoregressivePatchLearningDataset(
                data_dynamic=validation_data_dynamic,
                data_bc=validation_data_bc,
                data_patches=validation_data_patches,
                data_static=data_static,
                bc_generator=bc_generator,
                scaler=scaler,
                # Custom AR batching function
                ar_batch_fun=ar_batch_fun,
                training_mode=True,
                # Autoregressive settings
                input_k=input_k,
                output_k=output_k,
                forecast_cycle=forecast_cycle,
                ar_iterations=ar_scheduler.current_ar_iterations,
                stack_most_recent_prediction=stack_most_recent_prediction,
                # Timesteps
                subset_timesteps=(validation_data_patches.time.values if validation_data_patches is not None else None),
                # GPU settings
                device=device,
            )
        else:
            validation_ds = None
        print("- Creation of AutoregressiveDatasets: {:.0f}s".format(time.time() - t_i))

        ##--------------------------------------------------------------------.
        ## Create DataLoaders
        # - Prefetch (prefetch_factor*num_workers) batches parallelly into CPU
        # - At each AR iteration, the required data are transferred asynchronously to GPU
        # - If static data are provided, they are prefetched into the GPU
        # - Some data are duplicated in CPU memory because of the data overlap between forecast iterations.
        #   However this mainly affect boundary conditions data, because dynamic data
        #   after few AR iterations are the predictions of previous AR iteration.
        t_i = time.time()
        training_dl = AutoregressivePatchLearningDataLoader(
            dataset=training_ds,
            batch_size=training_batch_size,
            drop_last_batch=drop_last_batch,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            prefetch_in_gpu=prefetch_in_gpu,
            pin_memory=pin_memory,
            asyncronous_gpu_transfer=asyncronous_gpu_transfer,
            device=device,
        )
        if validation_data_dynamic is not None:
            validation_dl = AutoregressivePatchLearningDataLoader(
                dataset=validation_ds,
                batch_size=validation_batch_size,
                drop_last_batch=drop_last_batch,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                num_workers=num_workers,
                prefetch_in_gpu=prefetch_in_gpu,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                device=device,
            )
            validation_iterator = cylic_iterator(validation_dl)
            print(
                "- Creation of AutoregressiveDataLoaders: {:.0f}s".format(
                    time.time() - t_i
                )
            )
        else:
            validation_ds = None
            validation_iterator = None

        ##--------------------------------------------------------------------.
        # Initialize AR_TrainingInfo instance if not provided
        # - Initialization occurs when a new model training starts
        # - Passing an AR_TrainingInfo instance allows to continue model training from where it stopped !
        #   --> The ar_scheduler of previous training must be provided to ar_Training() !
        if ar_training_info is not None:
            if not isinstance(ar_training_info, AR_TrainingInfo):
                raise TypeError(
                    "If provided, 'ar_training_info' must be an instance of AR_TrainingInfo class."
                )
                # TODO: Check AR scheduler weights are compatible ! or need numpy conversion
                # ar_scheduler = ar_training_info.ar_scheduler
        else:
            ar_training_info = AR_TrainingInfo(
                ar_iterations=ar_iterations, epochs=epochs, ar_scheduler=ar_scheduler
            )

        ##--------------------------------------------------------------------.
        # Get dimension and feature infos
        # TODO: this is only used by the loss, --> future refactoring
        dim_info = training_ds.dim_info
        dim_order = training_ds.dim_order
        feature_info = training_ds.feature_info
        feature_order = training_ds.feature_order
        dim_info_dynamic = dim_info["dynamic"]
        # feature_dynamic = list(feature_info['dynamic'])

        ##--------------------------------------------------------------------.
        # Retrieve custom ar_batch_fun fuction
        ar_batch_fun = training_ds.ar_batch_fun
        ##--------------------------------------------------------------------.
        # Set model layers (i.e. batchnorm) in training mode
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ##--------------------------------------------------------------------.
        # Iterate along epochs
        print("")
        print("="*separator)
        flag_stop_training = False
        t_i_scoring = time.time()
        for epoch in range(epochs):
            ar_training_info.new_epoch()
            ##----------------------------------------------------------------.
            # Iterate along training batches
            training_iterator = iter(training_dl)
            ##----------------------------------------------------------------.
            # Compute collection points for SWAG training
            num_batches = len(training_iterator)
            batch_indices = range(num_batches)
            swag_training = swag and swag_model and epoch >= swa_start
            if swag_training:
                freq = int(num_batches / (swag_freq - 1))
                collection_indices = list(range(0, num_batches, freq))
            ##----------------------------------------------------------------.
            for batch_count in batch_indices:
                ##------------------------------------------------------------.
                # Retrieve the training batch
                training_batch_dict = next(training_iterator)
                ##------------------------------------------------------------.
                # Perform autoregressive training loop
                # - The number of AR iterations is determined by ar_scheduler.ar_weights
                # - If ar_weights are all zero after N forecast iteration:
                #   --> Load data just for F forecast iteration
                #   --> Autoregress model predictions just N times to save computing time
                dict_training_Y_predicted = {}
                dict_training_loss_per_ar_iteration = {}
                for ar_iteration in range(ar_scheduler.current_ar_iterations + 1):
                    # Retrieve X and Y for current AR iteration
                    # - ar_batch_fun() function stack together the required data from the previous AR iteration
                    torch_X, torch_Y = ar_batch_fun(
                        ar_iteration=ar_iteration,
                        batch_dict=training_batch_dict,
                        dict_Y_predicted=dict_training_Y_predicted,
                        asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                        device=device,
                    )
                    ##--------------------------------------------------------.
                    # # Print memory usage dataloader
                    # if device.type != 'cpu':
                    #     # torch.cuda.synchronize()
                    #     print("{}: {:.2f} MB".format(ar_iteration, torch.cuda.memory_allocated()/1000/1000))

                    ##--------------------------------------------------------.
                    # Forward pass and store output for stacking into next AR iterations
                    dict_training_Y_predicted[ar_iteration] = model(torch_X)

                    ##--------------------------------------------------------.
                    # Compute loss for current forecast iteration
                    # - The criterion expects [data_points, nodes, features]
                    # - Collapse all other dimensions to a 'data_points' dimension

                    # ordered_dynamic_variables_= [k for k, v in sorted(dim_info_dynamic.items(), key=lambda item: item[1])]
                    # Y_pred=Y_pred.rename(*ordered_dynamic_variables_)
                    # Y_obs=Y_obs.rename(*ordered_dynamic_variables_)
                    Y_pred, Y_obs = reshape_tensors_4_loss(
                        Y_pred=dict_training_Y_predicted[ar_iteration],
                        Y_obs=torch_Y,
                        dim_info_dynamic=dim_info_dynamic,
                        channels_first=channels_first
                    )
                    dict_training_loss_per_ar_iteration[ar_iteration] = criterion(
                        Y_obs, Y_pred
                    )

                    ##--------------------------------------------------------.
                    # If ar_training_strategy is "AR", perform backward pass at each AR iteration
                    if ar_training_strategy == "AR":
                        # - Detach gradient of Y_pred (to avoid RNN-style optimization)
                        dict_training_Y_predicted[
                            ar_iteration
                        ] = dict_training_Y_predicted[
                            ar_iteration
                        ].detach()  # TODO: should not be detached after backward?
                        # - AR weight the loss (aka weight sum the gradients ...)
                        current_ar_loss = dict_training_loss_per_ar_iteration[
                            ar_iteration
                        ]
                        current_ar_loss = (
                            current_ar_loss * ar_scheduler.ar_weights[ar_iteration]
                        )
                        # - Backpropagate to compute gradients (the derivative of the loss w.r.t. the parameters)
                        current_ar_loss.backward()
                        del current_ar_loss

                    ##--------------------------------------------------------.
                    # Remove unnecessary stored Y predictions
                    remove_unused_Y(
                        ar_iteration=ar_iteration,
                        dict_Y_predicted=dict_training_Y_predicted,
                        dict_Y_to_remove=training_batch_dict["dict_Y_to_remove"],
                    )

                    del Y_pred, Y_obs, torch_X, torch_Y
                    if ar_iteration == ar_scheduler.current_ar_iterations:
                        del dict_training_Y_predicted

                    ##--------------------------------------------------------.
                    # # Print memory usage dataloader + model
                    # if device.type != 'cpu':
                    #     torch.cuda.synchronize()
                    #     print("{}: {:.2f} MB".format(ar_iteration, torch.cuda.memory_allocated()/1000/1000))

                ##------------------------------------------------------------.
                # - Compute total (AR weighted) loss
                for i, (ar_iteration, loss) in enumerate(
                    dict_training_loss_per_ar_iteration.items()
                ):
                    if i == 0:
                        training_total_loss = (
                            ar_scheduler.ar_weights[ar_iteration] * loss
                        )
                    else:
                        training_total_loss += (
                            ar_scheduler.ar_weights[ar_iteration] * loss
                        )
                ##------------------------------------------------------------.
                # - If ar_training_strategy is RNN, perform backward pass after all AR iterations
                if ar_training_strategy == "RNN":
                    # - Perform backward pass using training_total_loss (after all AR iterations)
                    training_total_loss.backward()

                ##------------------------------------------------------------.
                # - Update the network weights
                optimizer.step()

                ##------------------------------------------------------------.
                # Zeros all the gradients for the next batch training
                # - By default gradients are accumulated in buffers (and not overwritten)
                optimizer.zero_grad(set_to_none=True)

                ##------------------------------------------------------------.
                # - Update training statistics             
                # TODO: This require CPU-GPU synchronization
                if ar_training_info.iteration_from_last_scoring == scoring_interval:
                    ar_training_info.update_training_stats(
                        total_loss=training_total_loss,
                        dict_loss_per_ar_iteration=dict_training_loss_per_ar_iteration,
                        ar_scheduler=ar_scheduler,
                        lr_scheduler=lr_scheduler,
                    )
                ##------------------------------------------------------------.
                # Printing infos (if no validation data available)
                # TODO: This require CPU-GPU synchronization
                if validation_ds is None:
                    if batch_count % scoring_interval == 0:
                        rounded_loss = round(dict_training_loss_per_ar_iteration[ar_iteration].item(), 5)
                        es_counter = early_stopping.counter
                        es_patience = early_stopping.patience
                        print(f"Epoch: {epoch} | Batch: {batch_count}/{num_batches} | "
                              f"AR: {ar_iteration} | Loss: {rounded_loss} | "
                              f"ES: {es_counter}/{es_patience}")
                    ##--------------------------------------------------------.
                    # The following code can be used to debug training if loss diverge to nan
                    if (dict_training_loss_per_ar_iteration[0].item() > 10000):
                        ar_training_info_fpath = os.path.join(os.path.dirname(model_fpath), "AR_TrainingInfo.pickle")
                        with open(ar_training_info_fpath, "wb") as handle:
                            pickle.dump(ar_training_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        raise ValueError("The training has diverged. "
                                         "The training info can be recovered using: \n"
                                         f"with open('{ar_training_info_fpath}', 'rb') as handle: \n"
                                         "    ar_training_info = pickle.load(handle)")

                ##------------------------------------------------------------.
                # TODO: SWAG Description
                if swag_training:
                    if batch_count in collection_indices:
                        swag_model.collect_model(model)

                ##------------------------------------------------------------.
                ### Run validation
                if validation_ds is not None:
                    if ar_training_info.iteration_from_last_scoring == scoring_interval:
                        # Set model layers (i.e. batchnorm) in evaluation mode
                        model.eval()

                        # Retrieve batch for validation
                        validation_batch_dict = next(validation_iterator)

                        # Initialize
                        dict_validation_loss_per_ar_iteration = {}
                        dict_validation_Y_predicted = {}

                        # ----------------------------------------------------.
                        # SWAG: collect, sample and update batch norm statistics
                        if swag_training:
                            swag_model.collect_model(model)
                            with torch.no_grad():
                                swag_model.sample(0.0)

                            bn_update_with_loader(
                                swag_model,
                                training_dl,
                                ar_iterations=ar_scheduler.current_ar_iterations,
                                asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                                device=device,
                            )

                        # ----------------------------------------------------.
                        # Disable gradient calculations
                        # - And do not update network weights
                        with torch.set_grad_enabled(False):
                            # Autoregressive loop
                            for ar_iteration in range(
                                ar_scheduler.current_ar_iterations + 1
                            ):
                                # Retrieve X and Y for current AR iteration
                                torch_X, torch_Y = ar_batch_fun(
                                    ar_iteration=ar_iteration,
                                    batch_dict=validation_batch_dict,
                                    dict_Y_predicted=dict_validation_Y_predicted,
                                    asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                                    device=device,
                                )

                                ##--------------------------------------------.
                                # Forward pass and store output for stacking into next AR iterations
                                if swag_training:
                                    dict_validation_Y_predicted[ar_iteration] = swag_model(torch_X)
                                else:
                                    dict_validation_Y_predicted[ar_iteration] = model(torch_X)

                                ##--------------------------------------------.
                                # Compute loss for current forecast iteration
                                # - The criterion expects [data_points, nodes, features]
                                # TODO: REFACTOR
                                Y_pred, Y_obs = reshape_tensors_4_loss(
                                    Y_pred=dict_validation_Y_predicted[ar_iteration],
                                    Y_obs=torch_Y,
                                    dim_info_dynamic=dim_info_dynamic,
                                    channels_first=channels_first
                                )
                                dict_validation_loss_per_ar_iteration[ar_iteration] = criterion(Y_obs, Y_pred)

                                ##--------------------------------------------.
                                # Remove unnecessary stored Y predictions
                                remove_unused_Y(
                                    ar_iteration=ar_iteration,
                                    dict_Y_predicted=dict_validation_Y_predicted,
                                    dict_Y_to_remove=validation_batch_dict["dict_Y_to_remove"],
                                )
                                del Y_pred, Y_obs, torch_X, torch_Y
                                if ar_iteration == ar_scheduler.current_ar_iterations:
                                    del dict_validation_Y_predicted

                        ##----------------------------------------------------.
                        ### Compute total (AR weighted) loss
                        for i, (ar_iteration, loss) in enumerate(
                            dict_validation_loss_per_ar_iteration.items()
                        ):
                            if i == 0:
                                validation_total_loss = (
                                    ar_scheduler.ar_weights[ar_iteration] * loss
                                )
                            else:
                                validation_total_loss += (
                                    ar_scheduler.ar_weights[ar_iteration] * loss
                                )

                        ##----------------------------------------------------.
                        ### Update validation info                                                                                        # TODO: This require CPU-GPU synchronization
                        ar_training_info.update_validation_stats(
                            total_loss=validation_total_loss,
                            dict_loss_per_ar_iteration=dict_validation_loss_per_ar_iteration,
                        )

                        ##----------------------------------------------------.
                        ### Reset model to training mode
                        model.train()

                        ##----------------------------------------------------.
                        ### Print scoring
                        t_f_scoring = round(time.time() - t_i_scoring)
                        print(
                            "Epoch: {} | Batch: {}/{} | AR: {} | Loss: {} | "
                            "ES: {}/{} | Elapsed time: {}s".format(
                                epoch,
                                batch_count,
                                num_batches,
                                ar_iteration,
                                round(
                                    dict_validation_loss_per_ar_iteration[
                                        ar_iteration
                                    ].item(),
                                    5,
                                ),  # TODO: This require CPU-GPU synchronization
                                early_stopping.counter,
                                early_stopping.patience,
                                t_f_scoring,
                            )
                        )
                        t_i_scoring = time.time()
                        ##----------------------------------------------------.
                        # The following code can be used to debug training if loss diverge to nan
                        if (
                            dict_validation_loss_per_ar_iteration[0].item() > 10000
                        ):  # TODO: This require CPU-GPU synchronization
                            ar_training_info_fpath = os.path.join(
                                os.path.dirname(model_fpath), "AR_TrainingInfo.pickle"
                            )
                            with open(ar_training_info_fpath, "wb") as handle:
                                pickle.dump(
                                    ar_training_info,
                                    handle,
                                    protocol=pickle.HIGHEST_PROTOCOL,
                                )
                            raise ValueError(
                                "The training has diverged. The training info can be recovered using: \n"
                                "with open({!r}, 'rb') as handle: \n"
                                "    ar_training_info = pickle.load(handle)".format(
                                    ar_training_info_fpath
                                )
                            )
                        ##----------------------------------------------------.

                ##------------------------------------------------------------.
                # - Update learning rate
                if lr_scheduler is not None:
                    lr_scheduler.step()

                ##------------------------------------------------------------.
                # - Update the AR weights
                ar_scheduler.step()

                ##------------------------------------------------------------.
                # - Evaluate stopping metrics and update AR scheduler if the loss has plateau
                if ar_training_info.iteration_from_last_scoring == scoring_interval:
                    # Reset counter for scoring
                    ar_training_info.reset_counter()
                    ##---------------------------------------------------------.
                    # If the model has not improved (based on early stopping settings)
                    # - If current_ar_iterations < ar_iterations --> Update AR scheduler
                    # - If current_ar_iterations = ar_iterations --> Stop training
                    if early_stopping is not None and early_stopping(ar_training_info):
                        # - If current_ar_iterations < ar_iterations --> Update AR scheduler
                        if ar_scheduler.current_ar_iterations < ar_iterations:
                            ##------------------------------------------------.
                            # Update the AR scheduler
                            ar_scheduler.update()
                            # Reset iteration counter from last AR weight update
                            ar_training_info.reset_iteration_from_last_ar_update()
                            # Reset early stopping
                            early_stopping.reset()
                            # Print info
                            current_ar_training_info = "(epoch: {}, iteration: {}, total_iteration: {})".format(
                                ar_training_info.epoch,
                                ar_training_info.epoch_iteration,
                                ar_training_info.iteration,
                            )
                            print("")
                            print("="*separator)
                            print("- Updating training to {} AR iterations {}.".format(
                                ar_scheduler.current_ar_iterations,
                                current_ar_training_info,
                                )
                            )
                            ##------------------------------------------------.
                            # Update Datasets (to prefetch the correct amount of data)
                            # - Training
                            del training_dl        # to avoid deadlocks
                            del training_iterator  # to avoid deadlocks
                            training_ds.update_ar_iterations(
                                ar_scheduler.current_ar_iterations
                            )
                            # - Validation
                            if validation_ds is not None:
                                del (
                                    validation_dl,
                                    validation_iterator,
                                )  # to avoid deadlocks
                                validation_ds.update_ar_iterations(
                                    ar_scheduler.current_ar_iterations
                                )
                            ##------------------------------------------------.
                            # Update DataLoaders (to prefetch the correct amount of data)
                            shuffle_seed += 1
                            training_dl = AutoregressivePatchLearningDataLoader(
                                dataset=training_ds,
                                batch_size=training_batch_size,
                                drop_last_batch=drop_last_batch,
                                shuffle=shuffle,
                                shuffle_seed=shuffle_seed,
                                num_workers=num_workers,
                                prefetch_factor=prefetch_factor,
                                prefetch_in_gpu=prefetch_in_gpu,
                                pin_memory=pin_memory,
                                asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                                device=device,
                            )
                            training_iterator = cylic_iterator(training_dl)
                            if validation_ds is not None:
                                validation_ds.update_ar_iterations(
                                    ar_scheduler.current_ar_iterations
                                )
                                validation_dl = AutoregressivePatchLearningDataLoader(
                                    dataset=validation_ds,
                                    batch_size=validation_batch_size,
                                    drop_last_batch=drop_last_batch,
                                    shuffle=shuffle,
                                    shuffle_seed=shuffle_seed,
                                    num_workers=num_workers,
                                    prefetch_in_gpu=prefetch_in_gpu,
                                    prefetch_factor=prefetch_factor,
                                    pin_memory=pin_memory,
                                    asyncronous_gpu_transfer=asyncronous_gpu_transfer,
                                    device=device,
                                )
                                validation_iterator = cylic_iterator(validation_dl)

                        ##----------------------------------------------------.
                        # - If current_ar_iterations = ar_iterations --> Stop training
                        else:
                            # Stop training
                            flag_stop_training = True
                            break

                ##------------------------------------------------------------.
                # - Update iteration count
                ar_training_info.step()

            ##----------------------------------------------------------------.
            ### Print epoch training statistics
            ar_training_info.print_epoch_info()

            if flag_stop_training:
                break
            ##----------------------------------------------------------------.
            # Option to save the model each epoch
            if save_model_each_epoch:
                if swag_training:
                    model_weights = swag_model.state_dict()
                else:
                    model_weights = model.state_dict()

                torch.save(model_weights, model_fpath[:-3] + "_epoch_{}".format(epoch) + ".h5")

        ##--------------------------------------------------------------------.
        ### Save final model weights
        if swag and swag_model:
            model_weights = swag_model.state_dict()
        else:
            model_weights = model.state_dict()

        torch.save(model_weights, f=model_fpath)

        ##--------------------------------------------------------------------.
        print(" ")
        print("=" * separator)
        print("- Training ended !")
        elapsed_hours = (time.time() - time_start_training) / 60 / 60
        print("- Total elapsed time: {:.2f} hours.".format(elapsed_hours))
        print("- Saving model to {}".format(model_fpath))

        ##--------------------------------------------------------------------.
        ### Save AR TrainingInfo
        print("=" * separator)
        print("- Saving training information")
        ar_info_fpath = os.path.join(os.path.dirname(model_fpath), "AR_TrainingInfo.pickle")
        with open(ar_info_fpath, "wb") as handle:
            pickle.dump(ar_training_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ##--------------------------------------------------------------------.
        ## Remove Dataset and DataLoaders to avoid deadlocks
        del validation_ds
        del validation_dl
        del validation_iterator
        del training_ds
        del training_dl
        del training_iterator
        ##--------------------------------------------------------------------.
        # Return training info object
        return ar_training_info

    # ------------------------------------------------------------------------.
    