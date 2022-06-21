#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:05:51 2021

@author: ghiggi
"""
import numpy as np
import torch
import torch.nn.functional as F

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


def transform_data_for_raft(data, feature_range=(-1, 1)):
    return data * (feature_range[1] - feature_range[0]) + feature_range[0]


def inverse_transform_data_for_raft(data, feature_range=(-1, 1)):
    return (data - feature_range[0])/(feature_range[1] - feature_range[0]) 


def pad_spatial_dim_for_raft(dim):
    nearest_multiple = np.ceil(dim/8) * 8
    diff = (nearest_multiple - dim)
    if diff % 2 == 0:
        pad_dim = [int(diff/2), int(diff/2)]
    else:
        pad_dim = [int(diff//2), int(diff//2+1)]

    return pad_dim


def check_data_for_raft(data, padding_mode="replicate"):
    y, x = data.shape[-2], data.shape[-1]
    pad = []
    if x % 8 != 0 and y % 8 != 0:
        pad.extend(pad_spatial_dim_for_raft(x))
        pad.extend(pad_spatial_dim_for_raft(y))
    elif x % 8 != 0 and y % 8 == 0:
        pad.extend(pad_spatial_dim_for_raft(x) + [0, 0])
    elif x % 8 == 0 and y % 8 != 0:
        pad.extend([0, 0] + pad_spatial_dim_for_raft(y))

    if len(pad) == 0:
        return data, pad
    else:
        return F.pad(data, tuple(pad), mode=padding_mode), pad


# https://mmediting.readthedocs.io/en/latest/_modules/mmedit/models/common/flow_warp.html
def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=x.dtype),
        torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output