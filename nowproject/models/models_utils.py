#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:05:51 2021

@author: ghiggi
"""
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

def check_skip_connection(skip_connection: str):
    """Check skip connection type."""
    if not isinstance(skip_connection, (str, type(None))):
        raise TypeError("'skip_connection' must be a string.")
    if skip_connection is None:
        skip_connection = "none"
    valid_options = ("none", "stack", "sum", "avg")
    if skip_connection not in valid_options:
        raise ValueError("'skip_connection' must be one of {}".format(valid_options))
    return skip_connection


def reshape_input_for_encoding(x: torch.Tensor, dim_names: List[str], output_shape: List[int]) -> torch.Tensor:
    """Reshapes the input data for the encoding where the last three dimensions should the spatio-temporal
    dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    dim_names : List[str]
        Input dimension names
    output_shape : List[int]
        Shape of the output tensor

    Returns
    -------
    torch.Tensor
        Reshaped tensor
    """
    x = (
        x.rename(*dim_names)
        .align_to("sample", "time", "y", "x", "feature")
        .rename(None)
    ) 
    x = x.reshape(output_shape)

    return x 

def reshape_input_for_decoding(x: torch.Tensor, dim_names: List[str], output_shape: List[int]) -> torch.Tensor:
    """Reshapes the input data after the decoding phase.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    dim_names : List[str]
        Output dimension names
    output_shape : List[int]
        Shape of the output tensor

    Returns
    -------
    torch.Tensor
        Reshaped tensor
    """
    x = x.reshape(output_shape) 
    x = (
        x.rename(*["sample", "time", "y", "x", "feature"])
        .align_to(*dim_names)
        .rename(None)
    )

    return x


def transform_data_for_raft(data: torch.Tensor, feature_range: Tuple[int, int] = (-1, 1)) -> torch.Tensor:
    """Transform the input data to the range expected by RAFT.

    Parameters
    ----------
    data : torch.Tensor
        Input data
    feature_range : tuple, optional
        The range to shift the data distribution to, by default (-1, 1)

    Returns
    -------
    torch.Tensor
        Shifted data
    """
    return data * (feature_range[1] - feature_range[0]) + feature_range[0]


def inverse_transform_data_for_raft(data: torch.Tensor, feature_range: Tuple[int, int] = (-1, 1)) -> torch.Tensor:
    """Transform the shift data back to its original range.

    Parameters
    ----------
    data : torch.Tensor
        Shifted data
    feature_range : Tuple[int, int], optional
        Feature range the data was shifted to, by default (-1, 1)

    Returns
    -------
    torch.Tensor
        Data shifted back to its original range
    """
    return (data - feature_range[0])/(feature_range[1] - feature_range[0]) 


def pad_spatial_dim_for_raft(dim: int) -> List[int]:
    """Computes the spatial padding of a certain dim of the 
    input data passed to RAFT.

    Parameters
    ----------
    dim : int
        Dimension size

    Returns
    -------
    List[int]
        Left and right padding of the input
    """
    nearest_multiple = np.ceil(dim/8) * 8
    diff = (nearest_multiple - dim)
    if diff % 2 == 0:
        pad_dim = [int(diff/2), int(diff/2)]
    else:
        pad_dim = [int(diff//2), int(diff//2+1)]

    return pad_dim


def check_data_for_raft(data: torch.Tensor, 
                        padding_mode: str = "replicate") -> Tuple[torch.Tensor, List[int]]:
    """Checks the input data shape before being passed to RAFT
    and pads the data if necessary. The size of the spatial dimensions 
    of the input must be a multiple of 8.

    Parameters
    ----------
    data : torch.Tensor
        Input data
    padding_mode : str, optional
        Padding mode, by default "replicate"

    Returns
    -------
    Tuple[torch.Tensor, List[int]]
        The processed input data and the padding applied to the spatial dimensions
    """
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
def flow_warp(x: torch.Tensor,
              flow: torch.Tensor,
              interpolation: str = 'bilinear',
              padding_mode: str = 'zeros',
              align_corners: bool = True) -> torch.Tensor:
    """Warp an image or a feature map with optical flow.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with size (n, c, h, w)
    flow : torch.Tensor
        Tensor with size (n, h, w, 2). The last dimension is
        a two-channel, denoting the width and height relative offsets.
        Note that the values are not normalized to [-1, 1].
    interpolation : str, optional
        Interpolation mode: 'nearest' or 'bilinear', by default 'bilinear'
    padding_mode : str, optional
        Padding mode: 'zeros' or 'border' or 'reflection', by default 'zeros'
    align_corners : bool, optional
        Whether to align corners, by default True

    Returns
    -------
    torch.Tensor
        Warped image or feature map.

    Raises
    ------
    ValueError
        The spatial sizes of the input must match those of the flow
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