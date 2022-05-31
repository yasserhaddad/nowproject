#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:05:51 2021

@author: ghiggi
"""
import numpy as np
from collections.abc import Iterable


def check_pool_method(pool_method):
    """Check valid pooling method."""
    # ('interp', 'maxval', 'maxarea', 'learn')  graph
    pool_method = pool_method.lower()
    return pool_method


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
