import os
import sys
import shutil
import argparse
from pathlib import Path

import time
import dask
import xarray as xr
import numpy as np
from torch import optim
from torchinfo import summary

from xforecasting.utils.io import get_ar_model_tensor_info
from xforecasting.utils.torch import summarize_model
from xforecasting import (
    AR_Scheduler,
    AutoregressivePredictions,
    rechunk_forecasts_for_verification,
    EarlyStopping,
)

from nowproject.utils.config import (
    read_config_file,
    write_config_file,
    get_model_settings,
    get_training_settings,
    get_ar_settings,
    get_dataloader_settings,
    get_pytorch_model,
    get_model_name,
    set_pytorch_settings,
    load_pretrained_model,
    create_experiment_directories,
    print_model_description,
    print_tensor_info,
    create_test_events_time_range
)

from xverif import xverif

# Project specific functions
from torch import nn
import nowproject.architectures as dl_architectures
from nowproject.loss import WeightedMSELoss, reshape_tensors_4_loss
from nowproject.training import AutoregressiveTraining
from nowproject.utils.plot import (
    plot_skill_maps, 
    plot_averaged_skill,
    plot_averaged_skills, 
    plot_skills_distribution
)
from nowproject.utils.scalers import RainScaler, RainBinScaler
from nowproject.data.data_config import METADATA
from nowproject.data.data_utils import load_static_topo_data

def main(cfg_path, data_dir_path, static_data_dir_path, test_events_path, 
         exp_dir_path, force=False):
    """General function for training models."""

    t_start = time.time()
    cfg = read_config_file(fpath=cfg_path)

    ##------------------------------------------------------------------------.
    # Load experiment-specific configuration settings
    model_settings = get_model_settings(cfg)
    ar_settings = get_ar_settings(cfg)
    training_settings = get_training_settings(cfg)
    dataloader_settings = get_dataloader_settings(cfg)

    ##------------------------------------------------------------------------.
    # Load Zarr Datasets
    data_dynamic = xr.open_zarr(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr")
    data_dynamic = data_dynamic.reset_coords(
        ["radar_names", "radar_quality", "radar_availability"], 
        drop=True
        )
    data_dynamic = data_dynamic.sel(time=slice(None, "2021-09-01T00:00"))
    # data_dynamic = data_dynamic.sel({"y": list(range(850, 450, -1)), "x": list(range(30, 320))})
    data_dynamic = data_dynamic.sel(
        {"y": list(range(835, 470, -1)), 
        "x": list(range(60, 300))}
        )
    data_dynamic = data_dynamic.rename({"precip": "feature"})[["feature"]]
    data_static = load_static_topo_data(static_data_dir_path, data_dynamic)
    data_bc = None

    ##------------------------------------------------------------------------.
    # Load scalers
    scaler = RainScaler(feature_min=np.log10(0.025), 
                        feature_max=np.log10(150), 
                        threshold=np.log10(0.1))


    ##------------------------------------------------------------------------.
    # Split data into train, test and validation set
    ## - Defining time split for training
    # training_years = np.array(["2018-05-01T00:00", "2020-12-31T23:57:30"], dtype="M8[s]")
    # validation_years = np.array(["2021-01-01T00:00", "2021-12-31T23:57:30"], dtype="M8[s]")
    training_years = np.array(["2018-10-01T00:00", "2018-10-10T23:57:30"], dtype="M8[s]")
    validation_years = np.array(["2021-01-01T00:00", "2021-01-10T23:57:30"], dtype="M8[s]")
    test_events = create_test_events_time_range(test_events_path)[:1]

    # - Split data sets
    t_i = time.time()
    training_data_dynamic = data_dynamic.sel(
        time=slice(training_years[0], training_years[-1])
    )
    validation_data_dynamic = data_dynamic.sel(
        time=slice(validation_years[0], validation_years[-1])
    )
    # test_data_dynamic = [data_dynamic.sel(time=event) for event in test_events]
    # test_data_dynamic = data_dynamic.sel(time=np.concatenate(test_events))

    print(
        "- Splitting data into train, validation and test sets: {:.2f}s".format(
            time.time() - t_i
        )
    )

    ##------------------------------------------------------------------------.
    # Define pyTorch settings (before PyTorch model definition)
    ## - Here inside eventually set the seed for fixing model weights initialization
    ## - Here inside the training precision is set (currently only float32 works)
    device = set_pytorch_settings(training_settings)


    ##------------------------------------------------------------------------.
    # Retrieve dimension info of input-output Torch Tensors
    tensor_info = get_ar_model_tensor_info(
        ar_settings=ar_settings,
        data_dynamic=training_data_dynamic,
        data_static=data_static,
        data_bc=None,
    )
    print_tensor_info(tensor_info)
    # - Add dim info to cfg file
    model_settings["tensor_info"] = tensor_info
    cfg["model_settings"]["tensor_info"] = tensor_info

    ##------------------------------------------------------------------------.
    # Print model settings
    print_model_description(cfg)
    
    ##------------------------------------------------------------------------.
    # Define the model architecture
    model = get_pytorch_model(module=dl_architectures, model_settings=model_settings)
   
    # If requested, load a pre-trained model for fine-tuning
    if model_settings["pretrained_model_name"] is not None:
        model_dir = exp_dir_path / model_settings["model_name"]
        load_pretrained_model(model=model, model_dir=model_dir.as_posix())

    # Transfer model to the device (i.e. GPU)
    model = model.to(device)

    # Summarize the model
    input_shape = tensor_info["input_shape"].copy()
    input_shape[0] = training_settings["training_batch_size"]
    print(
        summary(
            model, input_shape, col_names=["input_size", "output_size", "num_params"]
        )
    )

    _ = summarize_model(
        model=model,
        input_size=tuple(tensor_info["input_shape"][1:]),
        batch_size=training_settings["training_batch_size"],
        device=device,
    )

    # Generate the (new) model name and its directories
    if model_settings["model_name"] is not None:
        model_name = model_settings["model_name"]
    else:
        model_name = get_model_name(cfg)
        model_settings["model_name"] = model_name
        cfg["model_settings"]["model_name_prefix"] = None
        cfg["model_settings"]["model_name_suffix"] = None

    model_dir = create_experiment_directories(
        exp_dir=exp_dir_path, model_name=model_name, force=force
    )  # force=True will delete existing directory

    # Define model weights filepath
    model_fpath = model_dir / "model_weights" / "model.h5"

    ##------------------------------------------------------------------------.
    # Write config file in the experiment directory
    write_config_file(cfg=cfg, fpath=model_dir / "config.json")

    ##------------------------------------------------------------------------.
    # - Define custom loss function
    # --> TODO: For masking we could simply set weights to 0 !!!
    # criterion = WeightedMSELoss(weights=weights)
    criterion = WeightedMSELoss()

    ##------------------------------------------------------------------------.
    # - Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_settings["learning_rate"],
        eps=1e-7,
        weight_decay=0,
        amsgrad=False,
    )

    ##------------------------------------------------------------------------.
    # - Define AR_Weights_Scheduler
    ## - For RNN: growth and decay works well
    if training_settings["ar_training_strategy"] == "RNN":
        ar_scheduler = AR_Scheduler(
            method="LinearStep",
            factor=0.0005,
            fixed_ar_weights=[0],
            initial_ar_absolute_weights=[1, 1],
        )
    ## - FOR AR : Do not decay weights once they growthed
    elif training_settings["ar_training_strategy"] == "AR":
        ar_scheduler = AR_Scheduler(
            method="LinearStep",
            factor=0.0005,
            fixed_ar_weights=np.arange(0, ar_settings["ar_iterations"]),
            initial_ar_absolute_weights=[1, 1],
        )
    else:
        raise NotImplementedError(
            "'ar_training_strategy' must be either 'AR' or 'RNN'."
        )

    ##------------------------------------------------------------------------.
    # - Define Early Stopping
    ## - Used also to update ar_scheduler (aka increase AR iterations) if 'ar_iterations' not reached.
    patience = int(
        2000 / training_settings["scoring_interval"]
    )  # with 1000 and lr 0.005 crashed without AR update !
    minimum_iterations = 8000  # wtih 8000 worked
    minimum_improvement = 0.0001
    stopping_metric = "validation_total_loss"  # training_total_loss
    mode = "min"  # MSE best when low
    early_stopping = EarlyStopping(
        patience=patience,
        minimum_improvement=minimum_improvement,
        minimum_iterations=minimum_iterations,
        stopping_metric=stopping_metric,
        mode=mode,
    )

    ##------------------------------------------------------------------------.
    ### - Define LR_Scheduler
    lr_scheduler = None

    ##------------------------------------------------------------------------.
    ### - Train the model
    dask.config.set(
        scheduler="synchronous"
    )  # This is very important otherwise the dataloader hang

    ar_training_info = AutoregressiveTraining(
        model=model,
        model_fpath=model_fpath,
        # Loss settings
        criterion=criterion,
        reshape_tensors_4_loss=reshape_tensors_4_loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ar_scheduler=ar_scheduler,
        early_stopping=early_stopping,
        # Data
        data_static=data_static,
        training_data_dynamic=training_data_dynamic,
        training_data_bc=None,
        validation_data_dynamic=validation_data_dynamic,
        validation_data_bc=None,
        scaler=scaler,
        # Dataloader settings
        num_workers=dataloader_settings[
            "num_workers"
        ],  # dataloader_settings['num_workers'],
        prefetch_factor=dataloader_settings["prefetch_factor"],
        prefetch_in_gpu=dataloader_settings["prefetch_in_gpu"],
        drop_last_batch=dataloader_settings["drop_last_batch"],
        shuffle=dataloader_settings["random_shuffling"],
        shuffle_seed=training_settings["seed_random_shuffling"],
        pin_memory=dataloader_settings["pin_memory"],
        asyncronous_gpu_transfer=dataloader_settings["asyncronous_gpu_transfer"],
        # Autoregressive settings
        input_k=ar_settings["input_k"],
        output_k=ar_settings["output_k"],
        forecast_cycle=ar_settings["forecast_cycle"],
        ar_iterations=ar_settings["ar_iterations"],
        stack_most_recent_prediction=ar_settings["stack_most_recent_prediction"],
        # Training settings
        ar_training_strategy=training_settings["ar_training_strategy"],
        training_batch_size=training_settings["training_batch_size"],
        validation_batch_size=training_settings["validation_batch_size"],
        epochs=training_settings["epochs"],
        scoring_interval=training_settings["scoring_interval"],
        save_model_each_epoch=training_settings["save_model_each_epoch"],
        # GPU settings
        device=device,
    )

    ##------------------------------------------------------------------------.
    ### Create plots related to training evolution
    print("========================================================================================")
    print("- Creating plots to investigate training evolution")
    ar_training_info.plots(model_dir=model_dir, ylim=(0, 0.06))

    ##-------------------------------------------------------------------------.
    ### - Create predictions
    print("========================================================================================")
    print("- Running predictions")
    forecast_zarr_fpath = (
        model_dir / "model_predictions" / "forecast_chunked" / "test_forecasts.zarr"
    )
    if forecast_zarr_fpath.exists():
        shutil.rmtree(forecast_zarr_fpath)

    dask.config.set(scheduler="synchronous")  # This is very important otherwise the dataloader hang 
    # ds_forecasts = []
    # for i, event in enumerate(test_events):

    ds_forecasts = AutoregressivePredictions(
        model=model,
        forecast_reference_times=np.concatenate(test_events), 
        # Data
        data_dynamic=data_dynamic,
        data_static=data_static,
        data_bc=None,
        scaler_transform=scaler,
        scaler_inverse=scaler,
        # Dataloader options
        device=device,
        batch_size=50,  # number of forecasts per batch
        num_workers=dataloader_settings["num_workers"],
        prefetch_factor=dataloader_settings["prefetch_factor"],
        prefetch_in_gpu=dataloader_settings["prefetch_in_gpu"],
        pin_memory=dataloader_settings["pin_memory"],
        asyncronous_gpu_transfer=dataloader_settings["asyncronous_gpu_transfer"],
        # Autoregressive settings
        input_k=ar_settings["input_k"],
        output_k=ar_settings["output_k"],
        forecast_cycle=ar_settings["forecast_cycle"],
        stack_most_recent_prediction=ar_settings["stack_most_recent_prediction"],
        ar_iterations=20,  # How many time to autoregressive iterate
        # Save options
        zarr_fpath=forecast_zarr_fpath.as_posix(),  # None --> do not write to disk
        rounding=2,  # Default None. Accept also a dictionary
        compressor="auto",  # Accept also a dictionary per variable
        chunks="auto",
    )

    ds_forecasts = xr.open_zarr(forecast_zarr_fpath)
    ##------------------------------------------------------------------------.
    ### Reshape forecast Dataset for verification
    # - For efficient verification, data must be contiguous in time, but chunked over space (and leadtime)
    # - It also neeed to swap from 'forecast_reference_time' to the (forecasted) 'time' dimension
    #   The (forecasted) 'time'dimension is calculed as the 'forecast_reference_time'+'leadtime'
    print("========================================================================================")
    print("- Rechunk and reshape test set forecasts for verification")
    dask.config.set(scheduler="threads")
    t_i = time.time()
    verification_zarr_fpath = (
        model_dir / "model_predictions" / "space_chunked" / "test_forecasts.zarr"
    ).as_posix()

    if "time" in list(ds_forecasts.dims.keys()):
        ds_forecasts = ds_forecasts.drop_dims("time")

    # Check the chunk size of coords. If chunk size > coord shape, chunk size = coord shape.
    for coord in ds_forecasts.coords:
        if "chunks" in ds_forecasts[coord].encoding:
            new_chunks = []
            for i, c in enumerate(ds_forecasts[coord].encoding["chunks"]):
                if c >= ds_forecasts[coord].shape[i]:
                    new_chunks.append(ds_forecasts[coord].shape[i])
                else:
                    new_chunks.append(c)
            ds_forecasts[coord].encoding["chunks"] = tuple(new_chunks)


    ds_verification_format = rechunk_forecasts_for_verification(
        ds=ds_forecasts,
        chunks={'forecast_reference_time': -1, 'leadtime': 1, "x": 30, "y": 30},
        target_store=verification_zarr_fpath,
        max_mem="30GB",
    )
    print("   ---> Elapsed time: {:.1f} minutes ".format((time.time() - t_i) / 60))
    ##------------------------------------------------------------------------.
    ### - Run deterministic verification
    print("========================================================================================")
    print("- Run deterministic verification")
    # dask.config.set(scheduler='processes')
    # - Compute skills
    ds_skill = xverif.deterministic(
        pred=ds_verification_format.load().chunk({"x": 1, "y": 1}),
        obs=data_dynamic.sel(time=ds_verification_format.time).load().chunk({"x": 1, "y": 1}),
        forecast_type="continuous",
        aggregating_dim="time",
    )
    # - Save sptial skills
    ds_skill.to_netcdf((model_dir / "model_skills" / "deterministic_spatial_skill.nc"))

    ##------------------------------------------------------------------------.
    ### - Create verification summary plots and maps
    print("========================================================================================")
    print("- Create verification summary plots and maps")
    ds_averaged_skill = ds_skill.mean(dim=["y", "x"])
    
    # - Save averaged skills
    ds_averaged_skill.to_netcdf(model_dir / "model_skills" / "deterministic_global_skill.nc")

    # bbox = (451000, 30000, 850000, 319000)
    bbox = (470000, 60000, 835000, 300000)
    # - Create spatial maps
    plot_skill_maps(
        ds_skill=ds_skill,
        figs_dir=(model_dir / "figs" / "skills" / "SpatialSkill"),
        geodata=METADATA,
        bbox=bbox,
        skills=["BIAS", "RMSE", "rSD", "pearson_R2", "error_CoV"],
        variables=["feature"],
        suffix="",
        prefix="",
    )

    # - Create skill vs. leadtime plots
    plot_averaged_skill(ds_averaged_skill, skill="RMSE", variables=["feature"]).savefig(
        model_dir / "figs" / "skills" / "RMSE_skill.png"
    )
    plot_averaged_skills(ds_averaged_skill, variables=["feature"]).savefig(
        model_dir / "figs" / "skills" / "averaged_skill.png"
    )
    plot_skills_distribution(ds_skill, variables=["feature"]).savefig(
        model_dir / "figs" / "skills" / "skills_distribution.png",
    )

    ##-------------------------------------------------------------------------.
    print("========================================================================================")
    print(
        "- Model training and verification terminated. Elapsed time: {:.1f} hours ".format(
            (time.time() - t_start) / 60 / 60
        )
    )
    print("========================================================================================")
    ##-------------------------------------------------------------------------.

if __name__ == "__main__":
    default_data_dir = "/ltenas3/0_Data/NowProject/"
    default_static_data_dir = "/ltenas3/0_GIS/DEM Switzerland/"
    default_exp_dir = "/home/haddad/experiments/"
    default_config = "/home/haddad/nowproject/configs/UNet/AvgPool4-Conv3.json"
    default_test_events = "/home/haddad/nowproject/configs/events.json"

    parser = argparse.ArgumentParser(
        description="Training a numerical precipation nowcasting emulator"
    )
    parser.add_argument("--config_file", type=str, default=default_config)
    parser.add_argument("--test_events_file", type=str, default=default_test_events)
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--static_data_dir", type=str, default=default_static_data_dir)
    parser.add_argument("--exp_dir", type=str, default=default_exp_dir)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--force", type=str, default="True")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.force == "True":
        force = True
    else:
        force = False

    main(
        cfg_path=Path(args.config_file),
        exp_dir_path=Path(args.exp_dir),
        static_data_dir_path=Path(args.static_data_dir),
        test_events_path=Path(args.test_events_file),
        data_dir_path=Path(args.data_dir),
        force=force,
    )