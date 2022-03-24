#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:28:57 2021

@author: ghiggi
"""
import os
import sys

sys.path.append("../")

from nowproject.utils.config import (
    get_default_settings,
    read_config_file,
    write_config_file,
)

# - Config folder
config_path = "/home/haddad/NowProject/configs"

##----------------------------------------------------------------------------.
# Get default settings
cfg = get_default_settings()

# Current experiment (6h deltat)
cfg["ar_settings"]["input_k"] = [-5, -4, -3, -2, -1]
cfg["ar_settings"]["output_k"] = [0]
cfg["ar_settings"]["forecast_cycle"] = 1
cfg["ar_settings"]["ar_iterations"] = 6
cfg["ar_settings"]["stack_most_recent_prediction"] = True

## Training settings
cfg["training_settings"]["training_batch_size"] = 16
cfg["training_settings"]["validation_batch_size"] = 16
cfg["training_settings"]["epochs"] = 15
cfg["training_settings"]["numeric_precision"] = "float32"
cfg["training_settings"]["learning_rate"] = 0.007
cfg["training_settings"]["scoring_interval"] = 30
cfg["training_settings"]["save_model_each_epoch"] = False

# Reproducibility options
cfg["training_settings"]["deterministic_training"] = True
cfg["training_settings"]["seed_model_weights"] = 10
cfg["training_settings"]["seed_random_shuffling"] = 15

# GPU options
cfg["training_settings"]["gpu_training"] = True
cfg["training_settings"]["benchmark_cudnn"] = True
cfg["training_settings"]["dataparallel_training"] = False
cfg["training_settings"]["gpu_devices_ids"] = [0]

## Dataloader settings
cfg["dataloader_settings"]["random_shuffling"] = True
cfg["dataloader_settings"]["prefetch_in_gpu"] = False
cfg["dataloader_settings"]["prefetch_factor"] = 2
cfg["dataloader_settings"]["num_workers"] = 8
cfg["dataloader_settings"]["pin_memory"] = False
cfg["dataloader_settings"]["asyncronous_gpu_transfer"] = True
cfg["dataloader_settings"]["autotune_num_workers"] = False
cfg["dataloader_settings"]["drop_last_batch"] = False

##----------------------------------------------------------------------------.
### Create general configs for various architectecture

# Architecture options
kernel_size_conv = 3  # default is 3
kernel_size_pooling = 4  # default is 4

architecture_names = ["UNet", "ConvNet", "ResNet", "EPDNet"]
pool_methods = ["Max", "Avg"]

for architecture_name in architecture_names:
    for pool_method in pool_methods:
        custom_cfg = cfg
        custom_cfg["model_settings"]["architecture_name"] = architecture_name
        custom_cfg["model_settings"]["pool_method"] = pool_method
        custom_cfg["model_settings"]["kernel_size_conv"] = kernel_size_conv
        custom_cfg["model_settings"]["kernel_size_pooling"] = kernel_size_pooling

        # Create config directory
        tmp_dir = os.path.join(config_path, architecture_name)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        # Write config file
        tmp_config_name = (
            "-".join(
                [pool_method + "Pool" + kernel_size_pooling, "Conv" + kernel_size_conv]
            )
            + ".json"
        )
        write_config_file(custom_cfg, fpath=os.path.join(tmp_dir, tmp_config_name))