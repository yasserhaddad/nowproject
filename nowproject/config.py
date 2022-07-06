#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 07:50:15 2022

@author: ghiggi
"""
import os
import pathlib
import sys
import json
from typing import List
import torch
from torch import nn
import pickle
import shutil
import inspect
import types
import numpy as np
import datetime
import pytz as tz
import pandas as pd
import deepdiff

from xforecasting.utils.torch import (
    set_pytorch_deterministic,
    set_pytorch_numeric_precision,
)

# -----------------------------------------------------------------------------.
########################
### Default settings ###
########################

def get_default_model_settings():
    """Return some default settings for a NowProject model."""
    model_settings = {
        "pretrained_model_name": None,
        "model_name_prefix": None,
        "model_name": None,
        "model_name_suffix": None,
        # Architecture options,
        "last_layer_activation": False,
    }
    return model_settings


def get_default_training_settings():
    """Return some default settings for training the model."""
    training_settings = {
        "epochs": 15,
        "ar_training_strategy": "RNN",
        "learning_rate": 0.001,
        "training_batch_size": 8,
        "validation_batch_size": 8,
        "scoring_interval": 30,
        "save_model_each_epoch": False,
        "numeric_precision": "float32",
        "deterministic_training": False,
        "seed_model_weights": 100,
        "seed_random_shuffling": 120,
        "benchmark_cudnn": True,
        "gpu_training": True,
        "gpu_devices_ids": [0],
        "dataparallel_training": False,
    }
    return training_settings


def get_default_ar_settings():
    """Return some default settings for the autoregressive model."""
    ar_settings = {
        "input_k": [-4, -3, -2, -1],
        "output_k": [0],
        "forecast_cycle": 1,
        "ar_iterations": 6,
        "stack_most_recent_prediction": True,
    }
    return ar_settings


def get_default_dataloader_settings():
    """Return some default settings for the DataLoader."""
    dataloader_settings = {
        "random_shuffling": True,
        "drop_last_batch": True,
        "prefetch_in_gpu": False,
        "prefetch_factor": 5,
        "pin_memory": False,
        "asyncronous_gpu_transfer": True,
        "num_workers": 10,
        "autotune_num_workers": False,
    }
    return dataloader_settings


def get_default_SWAG_settings():
    """Return some default settings for the SWAG model."""
    dataloader_settings = {
        "SWAG": False,
        "target_learning_rate": 0.007,
        "no_cov_mat": False,
        "max_num_models": 40,
        "swag_freq": 10,
        "swa_start": 0,
        "sampling_scale": 0.1,
        "nb_samples": 10,
    }
    return dataloader_settings


def get_default_settings():
    """Return the default config settings."""
    ar_settings = get_default_ar_settings()
    training_settings = get_default_training_settings()
    model_settings = get_default_model_settings()
    dataloader_settings = get_default_dataloader_settings()
    default_settings = {
        "model_settings": model_settings,
        "dataloader_settings": dataloader_settings,
        "training_settings": training_settings,
        "ar_settings": ar_settings,
    }
    return default_settings


# -----------------------------------------------------------------------------.
########################
### I/O config file ####
########################
def read_config_file(fpath: str):
    """Create a dictionary of settings based on the json config file."""
    with open(fpath) as input_file:
        cfg = json.load(input_file)
    return cfg


def write_config_file(cfg: dict, fpath: str):
    """Write a json config file from the python dictionary config file."""
    with open(fpath, "w") as output_file:
        json.dump(cfg, output_file, indent=4)


# -----------------------------------------------------------------------------.
#############################
### Check config file keys ##
#############################
def get_model_settings(cfg: dict):
    """Return model settings from the config file."""
    # Initialize model settings
    model_settings = cfg["model_settings"]
    default_model_settings = get_default_model_settings()

    # Retrieve mandatory and optional keys
    mandatory_keys = [
        "architecture_name",
    ]
    optional_keys = list(default_model_settings.keys())

    # Retrieve mandatory model settings
    model_settings["architecture_name"] = cfg["model_settings"].get(
        "architecture_name", None
    )

    # Stop if some mandatory keys are missing
    flag_error = False
    for key in mandatory_keys:
        if model_settings[key] is None:
            flag_error = True
            print(
                "'{}' is a mandatory key that must be specified in the model settings section of the config file.".format(
                    key
                )
            )
    if flag_error:
        raise ValueError(
            "Specify the mandatory model settings keys in the config file!"
        )

    # Retrieve optional model settings
    for key in optional_keys:
        model_settings[key] = cfg["model_settings"].get(
            key, default_model_settings[key]
        )

    # Return model settings
    return model_settings


def get_training_settings(cfg: dict):
    """Return training settings from the config file."""
    # Initialize training settings
    training_settings = {}
    default_training_settings = get_default_training_settings()
    available_keys = list(default_training_settings.keys())

    # Check that only correct keys are specified
    cfg_keys = np.array(list(cfg["training_settings"].keys()))
    invalid_keys = cfg_keys[np.isin(cfg_keys, available_keys, invert=True)]
    if len(invalid_keys) > 0:
        for key in invalid_keys:
            print("'{}' is an unvalid training setting key!".format(key))
        raise ValueError(
            "Specify only correct training setting keys in the config file!"
        )

    # Retrieve optional training settings
    for key in available_keys:
        training_settings[key] = cfg["training_settings"].get(
            key, default_training_settings[key]
        )

    # Special checks
    if not isinstance(training_settings["gpu_devices_ids"], list):
        training_settings["gpu_devices_ids"] = [training_settings["gpu_devices_ids"]]

    if not training_settings["gpu_training"]:
        if training_settings["dataparallel_training"]:
            print("DataParallel training is available only on GPUs!")
            training_settings["dataparallel_training"] = False

    # Return training settings
    return training_settings


def get_dataloader_settings(cfg: dict):
    """Return dataloader settings from the config file."""
    # Initialize dataloader settings
    dataloader_settings = {}
    default_dataloader_settings = get_default_dataloader_settings()
    available_keys = list(default_dataloader_settings.keys())

    # Check that only correct keys are specified
    cfg_keys = np.array(list(cfg["dataloader_settings"].keys()))
    invalid_keys = cfg_keys[np.isin(cfg_keys, available_keys, invert=True)]
    if len(invalid_keys) > 0:
        for key in invalid_keys:
            print("'{}' is an unvalid dataloader setting key!".format(key))
        raise ValueError(
            "Specify only correct dataloader setting keys in the config file!"
        )

    # Retrieve optional dataloader settings
    for key in available_keys:
        dataloader_settings[key] = cfg["dataloader_settings"].get(
            key, default_dataloader_settings[key]
        )

    # Return dataloader settings
    return dataloader_settings


def get_ar_settings(cfg: dict):
    """Return AR settings from the config file."""
    # Initialize AR settings
    ar_settings = {}
    default_ar_settings = get_default_ar_settings()
    available_keys = list(default_ar_settings.keys())

    # Check that only correct keys are specified
    cfg_keys = np.array(list(cfg["ar_settings"].keys()))
    invalid_keys = cfg_keys[np.isin(cfg_keys, available_keys, invert=True)]
    if len(invalid_keys) > 0:
        for key in invalid_keys:
            print("'{}' is an unvalid AR setting key!".format(key))
        raise ValueError("Specify only correct AR setting keys in the config file!")

    # Retrieve optional AR settings
    for key in available_keys:
        ar_settings[key] = cfg["ar_settings"].get(key, default_ar_settings[key])

    # Ensure input_k and output_k are list
    if not isinstance(ar_settings["input_k"], list):
        ar_settings["input_k"] = [ar_settings["input_k"]]
    if not isinstance(ar_settings["output_k"], list):
        ar_settings["output_k"] = [ar_settings["output_k"]]

    # Return AR settings
    return ar_settings


def get_SWAG_settings(cfg: dict):
    """Return SWAG settings from the config file."""
    # Initialize AR settings
    SWAG_settings = {}
    default_SWAG_settings = get_default_SWAG_settings()
    available_keys = list(default_SWAG_settings.keys())

    # Check that only correct keys are specified
    cfg_keys = np.array(list(cfg["SWAG_settings"].keys()))
    invalid_keys = cfg_keys[np.isin(cfg_keys, available_keys, invert=True)]
    if len(invalid_keys) > 0:
        for key in invalid_keys:
            print("'{}' is an unvalid SWAG setting key!".format(key))
        raise ValueError("Specify only correct SWAG setting keys in the config file!")

    # Retrieve optional AR settings
    for key in available_keys:
        SWAG_settings[key] = cfg["SWAG_settings"].get(key, default_SWAG_settings[key])

    # Return AR settings
    return SWAG_settings


def check_same_dict(x, y):
    ddif = deepdiff.DeepDiff(x, y, ignore_type_in_groups=[(str, np.str_)])
    if len(ddif) > 0:
        print("The two dictionaries have the following mismatches :")
        print(ddif)
        raise ValueError("Not same dictionary.")
    return None


# -----------------------------------------------------------------------------.
#################################
### Checks config key values ####
#################################
def check_numeric_precision(numeric_precision: str):
    """Check numeric precision argument."""
    if not isinstance(numeric_precision, str):
        raise TypeError("Specify 'numeric_precision' as a string")
    if not [numeric_precision] in ["float64", "float32", "float16", "bfloat16"]:
        raise ValueError(
            "Valid 'numeric precision' are: 'float64', 'float32','float16' and 'bfloat16'"
        )


# -----------------------------------------------------------------------------.
########################
### Model definition ###
########################
def get_pytorch_model(module: nn.Module, model_settings: dict):
    """
    Define a NowProject model based on model_settings configs.

    The architecture structure must be defined in the 'architectures' custom python file

    Parameters
    ----------
    module : module
        Imported python module containing the architecture definition.
    model_settings : dict
        Dictionary containing all architecture options.
    """
    if not isinstance(module, types.ModuleType):
        raise TypeError(
            "'module' must be a preimported module with the architecture definition."
        )
    # - Retrieve the required model arguments
    NowProjectModelClass = getattr(module, model_settings["architecture_name"])
    fun_args = inspect.getfullargspec(NowProjectModelClass.__init__).args
    model_args = {k: model_settings[k] for k in model_settings.keys() if k in fun_args}
    # - Define NowProjectmodel
    model = NowProjectModelClass(**model_args)
    return model


def get_pytorch_SWAG_model(module: nn.Module, model_settings: dict, swag_settings):
    """
    Define a NowProject SWAG model based on model_settings and swag configs.

    The architecture structure must be define in the 'module' custom python file

    Parameters
    ----------
    module : module
        Imported python module containing the architecture definition.
    model_settings : dict
        Dictionary containing all architecture options.
    swag_settings : dict
        Dictionary containing all the SWAG options.
    """
    from modules.swag import SWAG

    if not isinstance(module, types.ModuleType):
        raise TypeError(
            "'module' must be a preimported module with the architecture definition."
        )
    # - Retrieve the required model arguments
    NowProjectModelClass = getattr(module, model_settings["architecture_name"])
    fun_args = inspect.getfullargspec(NowProjectModelClass.__init__).args
    model_args = {k: model_settings[k] for k in model_settings.keys() if k in fun_args}
    # - Define NowProjectSWAG model
    swag_model = SWAG(
        NowProjectModelClass,
        no_cov_mat=swag_settings["no_cov_mat"],
        max_num_models=swag_settings["max_num_models"],
        **model_args
    )
    return swag_model


##----------------------------------------------------------------------------.
def load_pretrained_model(model: nn.Module, model_dir: str, strict: bool = True):
    """Load a pre-trained pytorch model using HDF5 saved weights."""
    model_fpath = os.path.join(model_dir, "model_weights", "model.h5")
    state = torch.load(model_fpath)
    model.load_state_dict(state, strict=strict)


def load_pretrained_ar_scheduler(exp_dir: str, model_name: str):
    """Load a pre-trained AR scheduler."""
    training_info_fpath = os.path.join(
        exp_dir, model_name, "training_info", "ar_TrainingInfo.pickle"
    )
    with open(training_info_fpath, "rb") as f:
        training_info = pickle.load(f)
    ar_scheduler = pickle.loads(training_info.ar_scheduler)

    return ar_scheduler


# -----------------------------------------------------------------------------.
#########################
### Pytorch settings ####
#########################
def set_pytorch_settings(training_settings: dict):
    """Set training options with pytorch."""
    # Retrieve pytorch settings options
    deterministic_training = training_settings["deterministic_training"]
    seed_model_weights = training_settings["seed_model_weights"]
    benchmark_cudnn = training_settings["benchmark_cudnn"]
    gpu_training = training_settings["gpu_training"]
    gpu_devices_ids = training_settings["gpu_devices_ids"]
    numeric_precision = training_settings["numeric_precision"]

    ##------------------------------------------------------------------------.
    # Set options for deterministic training
    if deterministic_training:
        set_pytorch_deterministic(seed=seed_model_weights)

    ##------------------------------------------------------------------------.
    # If requested, autotunes to the best cuDNN kernel (for performing convolutions)
    # --> Find the best algorithm to use with the available hardware.
    # --> Usually leads to faster runtime.
    if benchmark_cudnn and not deterministic_training:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
    ##------------------------------------------------------------------------.
    # Return the device to make the pytorch architecture working both on CPU and GPU
    if gpu_training:
        if torch.cuda.is_available():
            device = torch.device(gpu_devices_ids[0])
        else:
            print("- GPU is not available. Switching to CPU !")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # ------------------------------------------------------------------------.
    # Set numeric precision
    set_pytorch_numeric_precision(numeric_precision=numeric_precision, device=device)

    # ------------------------------------------------------------------------.
    # Return the torch device
    return device


# -----------------------------------------------------------------------------.
############################
### Experiment structure ###
############################
def get_model_name(cfg: dict):
    """Create a model name based on the config settings."""
    ##------------------------------------------------------------------------.
    # Retrieve model_name, suffix and prefix
    model_name = cfg["model_settings"]["model_name"]
    model_name_prefix = cfg["model_settings"]["model_name_prefix"]
    model_name_suffix = cfg["model_settings"]["model_name_suffix"]

    ##------------------------------------------------------------------------.
    # Define model name based on config settings if not specified
    if model_name is None:
        # Retrieve important "discriminatory" settings
        architecture_name = cfg["model_settings"]["architecture_name"]
        pool_method = cfg["model_settings"]["pool_method"]
        ar_training_strategy = cfg["training_settings"]["ar_training_strategy"]
        ar_iterations = cfg["ar_settings"]["ar_iterations"]
        # Create model name
        model_name = "-".join(
            [
                ar_training_strategy,
                "AR" + str(ar_iterations),
                architecture_name,
                pool_method + "Pooling",
            ]
        )
        # model_name = model_name[:-1] # remove last "-"
    ##------------------------------------------------------------------------.
    # Add prefix and suffix if specified
    if model_name_prefix is not None:
        model_name = "-".join([model_name_prefix, model_name])
    if model_name_suffix is not None:
        model_name = "-".join([model_name, model_name_suffix])

    ##------------------------------------------------------------------------.
    # Update cfg model_name
    cfg["model_settings"]["model_name"] = model_name

    ##------------------------------------------------------------------------.
    # Return model name
    return model_name


def create_experiment_directories(exp_dir: pathlib.Path, model_name: str, suffix: str = "", 
                                  force: bool = False):
    """Create the required directory for a specific NowProject model."""
    # Check if the experiment directory already exists
    dir_name = f"{model_name}-{suffix}" if suffix != "" else model_name
    exp_dir = exp_dir / dir_name
    if exp_dir.exists():
        if force:
            shutil.rmtree(exp_dir)
        else:
            raise ValueError(
                "The directory {} already exists.\
                             force=True in create_experiment_directories() will delete content of the existing directory.\
                             Please delete such directory manually or: \
                                 - specify 'model_name' in model_settings \
                                 - specify 'model_prefix' and/or 'model_suffix' in model_settings".format(
                    exp_dir
                )
            )

    ##------------------------------------------------------------------------.
    # Define standard directories
    model_weights_dir = exp_dir / "model_weights"
    figures_dir = exp_dir / "figs"
    figs_skills_dir = figures_dir / "skills"
    figs_training_info_dir = figures_dir / "training_info"
    model_predictions_dir = exp_dir / "model_predictions"
    space_chunked_forecasts_dir = model_predictions_dir / "space_chunked"
    forecast_chunked_forecasts_dir = model_predictions_dir / "forecast_chunked"
    model_skills_dir = exp_dir / "model_skills"

    ##------------------------------------------------------------------------.
    # Create directories
    model_weights_dir.mkdir(parents=True, exist_ok=False)
    figs_skills_dir.mkdir(parents=True, exist_ok=False)
    figs_training_info_dir.mkdir(parents=True, exist_ok=False)
    model_skills_dir.mkdir(parents=True, exist_ok=False)
    space_chunked_forecasts_dir.mkdir(parents=True, exist_ok=False)
    forecast_chunked_forecasts_dir.mkdir(parents=True, exist_ok=False)

    ##------------------------------------------------------------------------.
    # Return the experiment directory
    return exp_dir


##------------------------------------------------------------------------.
#########################
### Print model info ####
#########################
def pretty_printing(d: dict, indent: int = 0, indent_factor: int = 2):
    """Pretty pritting of nested dictionaries."""
    for key, value in d.items():
        print((" " * indent * indent_factor) + "- " + str(key) + ":", end="")
        if isinstance(value, dict):
            print(end="\n")
            pretty_printing(value, indent=indent + 1, indent_factor=indent_factor)
        else:
            print(" " + str(value), end="\n")


def print_tensor_info(tensor_info: dict):
    """Pretty printing of tensor dimension information."""
    print("- Input-Output Tensors characteristics:")
    pretty_printing(tensor_info, indent=1, indent_factor=2)


def print_model_description(cfg: dict, dim_info: dict = None):
    """Pretty printing of experiment settings."""
    print("- Experiment settings:")
    if dim_info is not None:
        cfg["dim_info"] = dim_info
    pretty_printing(cfg, indent=1, indent_factor=2)

##------------------------------------------------------------------------.
####################
### Test events ####
####################

def create_event_time_range(event_dict: dict, freq: str = "2min30s") -> np.ndarray:
    """Creates a time range from the information contained in the event_dict.

    Parameters
    ----------
    event_dict : dict
        Dictionary with keys "start_time", "duration" and "timezone"
    freq : str, optional
        Timestep interval, by default "2min30s"

    Returns
    -------
    np.ndarray
        Time range starting at the indicated time in event_dict for the given
        duration and timestep.
    """
    start_time = pd.to_datetime(event_dict["start_time"])
    return pd.date_range(start=start_time, 
                         end=start_time + datetime.timedelta(hours=event_dict["duration"]), 
                         freq=freq, tz=event_dict["timezone"]).to_numpy().astype("M8[s]")

def create_event_time_range_autoregressive(event_dict: dict, nb_ar_iterations: int, 
                                           freq: str = "2min30s") -> np.ndarray:
    """Creates an autoregressive time range with nb_ar_iterations additional timesteps
    preceding the indicated start time.

    Parameters
    ----------
    event_dict : dict
        Dictionary with keys "start_time", "duration" and "timezone"
    nb_ar_iterations : int
        Number of Autoregressive Iterations
    freq : str, optional
        Timestep interval, by default "2min30s"

    Returns
    -------
    np.ndarray
        Autoregressive time range starting with nb_ar_iterations additional timesteps
        preceding the indicated start time, for the given duration and timestep.
    """
    regular_time_range = create_event_time_range(event_dict, freq=freq)
    additional_time_range = pd.date_range(end=regular_time_range[0], periods=nb_ar_iterations + 1, 
                                          freq=freq, tz=event_dict["timezone"], 
                                          closed="left").to_numpy().astype("M8[s]")
    return np.concatenate([additional_time_range, regular_time_range])

def create_test_events_autoregressive_time_range(fpath: str, nb_ar_iterations: int, 
                                                freq: str = "2min30s") -> List[np.ndarray]:
    """Creates test events autoregressive time ranges with nb_ar_iterations additional timesteps
    preceding the indicated start time.

    Parameters
    ----------
    fpath : str
        Path to the JSON file containing the event dictionaries
    nb_ar_iterations : int
        Number of Autoregressive Iterations
    freq : str, optional
        Timestep interval, by default "2min30s"

    Returns
    -------
    List[np.ndarray]
        List of autoregressive time ranges
    """
    with open(fpath, "r", encoding="utf-8") as f:
        event_dicts = json.load(f)
        return [create_event_time_range_autoregressive(d, nb_ar_iterations, freq=freq) for d in event_dicts]

def create_test_events_time_range(fpath: str, freq: str = "2min30s") -> List[np.ndarray]:
    """Creates test events time ranges starting at start_time for a given duration
    and timestep.

    Parameters
    ----------
    fpath : str
        Path to the JSON file containing the event dictionaries
    freq : str, optional
        Timestep interval, by default "2min30s"

    Returns
    -------
    List[np.ndarray]
        List of time ranges
    """
    with open(fpath, "r", encoding="utf-8") as f:
        event_dicts = json.load(f)
        return [create_event_time_range(d, freq=freq) for d in event_dicts]
