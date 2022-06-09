#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:05:51 2021

@author: ghiggi
"""
import numpy as np

def check_skip_connection(skip_connection):
    """Check skip connection type."""
    if not isinstance(skip_connection, (str, type(None))):
        raise TypeError("'skip_connection' must be a string.")
    if skip_connection is None:
        skip_connection = "none"
    valid_options = ("none", "stack", "sum", "avg")
    if skip_connection not in valid_options:
        raise ValueError("'skip_connection' must be one of {}".format(valid_options))
    return skip_connection


def reshape_input_for_encoding(x, dim_names, output_shape):
    x = (
        x.rename(*dim_names)
        .align_to("sample", "time", "y", "x", "feature")
        .rename(None)
    ) 
    x = x.reshape(output_shape)

    return x 

def reshape_input_for_decoding(x, dim_names, output_shape):
    x = x.reshape(output_shape) 
    x = (
        x.rename(*["sample", "time", "y", "x", "feature"])
        .align_to(*dim_names)
        .rename(None)
    )

    return x