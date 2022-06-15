from typing import List, Union, Callable
import xarray as xr
import numpy as np
from xscaler.checks import (
    check_variable_dim,
    check_rename_dict,
    get_xarray_variables,
)

from pysteps.utils import transformation

# Transform and inverse transform functions

def normalize_transform(da: xr.DataArray, threshold: float, feature_min: float,
                        feature_max: float):
    """Normalize the data.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    threshold : float
        Threshold below which the values will be set to feature_min
    feature_min : float
        Minimal desired value of the data
    feature_max : float
        Maximal desired value of the data

    Returns
    -------
    xr.DataArray
        Result of the Log-Normalize transform
    """
    da = da.where(da >= threshold, feature_min)
    da = da.clip(max=feature_max)
    da = (da - feature_min) / (feature_max - feature_min)
    da = da.fillna(0.0)

    return da

def normalize_inverse_transform(da: xr.DataArray, threshold: float, feature_min: float,
                                feature_max: float):
    """Revert the normalization of the data.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    threshold : float
        Threshold below which the values will be set to feature_min
    feature_min : float
        Minimal desired value of the data
    feature_max : float
        Maximal desired value of the data

    Returns
    -------
    xr.DataArray
        Result of the inverse of the Log-Normalize transform
    """
    da = da * (feature_max - feature_min) + feature_min
    da = da.where(da >= threshold, feature_min)

    return da


def log_normalize_transform(da: xr.DataArray, threshold: float, feature_min: float, 
                            feature_max: float) -> xr.DataArray:
    """Apply the Log-Normalize transform.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    threshold : float
        Threshold below which the values will be set to feature_min
    feature_min : float
        Minimal desired value of the data
    feature_max : float
        Maximal desired value of the data

    Returns
    -------
    xr.DataArray
        Result of the Log-Normalize transform
    """
    da = xr.ufuncs.log10(da + 0.0001)
    da = normalize_transform(da, threshold, feature_min, feature_max)

    return da


def log_normalize_inverse_transform(da: xr.DataArray, threshold: float, feature_min: float, 
                                    feature_max: float) -> xr.DataArray:
    """Apply the inverse of the Log-Normalize transform.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    threshold : float
        Threshold below which the values will be set to feature_min
    feature_min : float
        Minimal desired value of the data
    feature_max : float
        Maximal desired value of the data

    Returns
    -------
    xr.DataArray
        Result of the inverse of the Log-Normalize transform
    """
    da = normalize_inverse_transform(da, threshold, feature_min, feature_max)
    da = 10 ** da
    
    return da   


def bin_transform(da: xr.DataArray, bins: List[float]) -> xr.DataArray:
    """Apply the bin transform.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    bins : List[float]
        Bins to place values in

    Returns
    -------
    xr.DataArray
        Result of the bin transform
    """
    da = da.clip(max=(bins[-1] - 0.01))
    da = xr.apply_ufunc(np.digitize, 
                        da, 
                        kwargs={"bins": bins}, 
                        dask="parallelized",
                        output_dtypes=['i8'])
    return da


def bin_inverse_transform(da: xr.DataArray, inverse_bins: Callable):
    """Apply the inverse bin transform.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    inverse_bins : Callable
        Function that will convert binned values back to values
        in the original data range.
        Example : each bin value can be mapped to the centre of
        the map.

    Returns
    -------
    xr.DataArray
        Result of the inverse bin transform
    """
    da = xr.ufuncs.rint(da).astype(int)
    da = xr.apply_ufunc(inverse_bins, da, dask='parallelized', output_dtypes=[float])        
    return da


def dB_transform(da: xr.DataArray, threshold: float, zero_value: float) -> xr.DataArray:
    """Apply the dB transform, converting the array from rainrate (mm/h) to dB.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    threshold : float
        Threshold below which the values be converted
        to the zero_value
    zero_value : float
        Value to set for values below the threshold
    
    Returns
    -------
    xr.DataArray
        Result of the dB transform
    """
    def dB_transformation(values, threshold, zero_value):
        return transformation.dB_transform(values, threshold=threshold, zerovalue=zero_value)[0]

    da = xr.apply_ufunc(dB_transformation, 
                        da, 
                        kwargs={"threshold": threshold, "zero_value": zero_value}, 
                        dask="parallelized",
                        output_dtypes=['float'])
    da = da.where(xr.ufuncs.isfinite(da), zero_value)
    return da


def dB_inverse_transform(da: xr.DataArray, inverse_threshold: float) -> xr.DataArray:
    """Apply the dB inverse transform, converting the array from dB to rainrate (mm/h).

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    inverse_threshold : float
        Threshold below which the values will be converted to 0
    
    Returns
    -------
    xr.DataArray
        Result of the dB inverse transform
    """
    def dB_inverse_transformation(values, inverse_threshold):
        return transformation.dB_transform(values, threshold=inverse_threshold, inverse=True)[0]

    da = xr.apply_ufunc(dB_inverse_transformation, 
                        da, 
                        kwargs={"inverse_threshold": inverse_threshold},
                        dask="parallelized",
                        output_dtypes=['float'])
    return da


def log_epsilon_transform(da: xr.DataArray, epsilon: float) -> xr.DataArray:
    """Apply the Log-Epsilon transform.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    epsilon : float
        Epsilon value to divide the array with

    Returns
    -------
    xr.DataArray
        Result of the Log-Epsilon transform
    """
    return xr.ufuncs.log10(1 + da/epsilon)


def log_epsilon_inverse_transform(da: xr.DataArray, epsilon: float, 
                                  round_decimals: bool = True) -> xr.DataArray:
    """Apply the inverse transform of the Log-Epsilon transform.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to transform
    epsilon : float
        Epsilon value to multiply the log-inverse 
    round_decimals : bool, optional
        If true, round the result to 2 decimals, by default True

    Returns
    -------
    xr.DataArray
        Result of the inverse transform of the Log-Epsilon transform 
    """
    da = (10**da - 1) * epsilon
    if round_decimals:
        da = da.round(2)
    return da

# Scaler class

class Scaler:
    def __init__(self, 
                 fn_transform: Callable, 
                 fn_inverse_transform: Callable,
                 transform_kwargs: dict,
                 inverse_transform_kwargs: dict):
        if not callable(fn_transform):
            raise TypeError("'fn_transform' should be a function.")
        if not callable(fn_inverse_transform):
            raise TypeError("'fn_inverse_transform' should be a function.")
        if not isinstance(transform_kwargs, dict):
            raise TypeError("'transform_kwargs' should be a dict.")
        if not isinstance(inverse_transform_kwargs, dict):
            raise TypeError("'inverse_transform_kwargs' should be a dict.")

        self.fn_transform = fn_transform
        self.fn_inverse_transform = fn_inverse_transform
        self.transform_kwargs = transform_kwargs
        self.inverse_transform_kwargs = inverse_transform_kwargs
    
    #------------------------------------------------------------------------.
    def transform(self, new_data, variable_dim=None, rename_dict=None):
        """Transform data"""
        ##--------------------------------------------------------------------.
        # Get variables to transform
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim=variable_dim, data=new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        new_data_dims = list(new_data.dims.keys())
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)
            # Create dictionary for resetting dimensions name as original
            inv_rename_dict = {v: k for k, v in rename_dict.items()}
            # Rename dimensions
            new_data = new_data.rename(rename_dict)
        ##--------------------------------------------------------------------.
        ## Transform variables
        new_data = new_data.copy()
        for var in data_vars:
            new_data[var] = self.fn_transform(new_data[var], **self.transform_kwargs)
        ##--------------------------------------------------------------------.
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed:
            new_data = new_data.rename(inv_rename_dict)
        ##--------------------------------------------------------------------.
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return (
                    new_data.to_array(dim="variable", name=da_name)
                    .squeeze()
                    .drop("variable")
                    .transpose(*da_dims_order)
                )
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(
                    *da_dims_order
                )
        else:
            return new_data

    ##------------------------------------------------------------------------.
    def inverse_transform(self, new_data, variable_dim=None, rename_dict=None):
        """Inverse transforn rain data from normalized log scale"""
        ##--------------------------------------------------------------------.
        # Get variables to transform
        data_vars = get_xarray_variables(new_data, variable_dim=variable_dim)
        ##--------------------------------------------------------------------.
        # If input is DataArray --> Convert to Dataset
        flag_DataArray = False
        if isinstance(new_data, xr.DataArray):
            flag_DataArray = True
            da_name = new_data.name
            da_dims_order = new_data.dims
            variable_dim = check_variable_dim(variable_dim=variable_dim, data=new_data)
            new_data = new_data.to_dataset(dim=variable_dim)
        
        ##--------------------------------------------------------------------.
        # Rename new_data dimensions if rename_dict is provided
        flag_dim_renamed = False
        if rename_dict is not None:
            flag_dim_renamed = True
            # Check rename_dict (ensure {from:to} format )
            rename_dict = check_rename_dict(data=new_data, rename_dict=rename_dict)
            # Create dictionary for resetting dimensions name as original
            inv_rename_dict = {v: k for k, v in rename_dict.items()}
            # Rename dimensions
            new_data = new_data.rename(rename_dict)
        ##--------------------------------------------------------------------.
        ## Transform variables
        new_data = new_data.copy()
        for var in data_vars:
            new_data[var] = self.fn_inverse_transform(new_data[var], 
                                                      **self.inverse_transform_kwargs)
        ##--------------------------------------------------------------------.
        # Rename dimension as new_data (if necessary)
        if flag_dim_renamed:
            new_data = new_data.rename(inv_rename_dict)
        ##--------------------------------------------------------------------.
        # Reshape to DataArray if new_data was a DataArray
        if flag_DataArray:
            if variable_dim is None:
                return (
                    new_data.to_array(dim="variable", name=da_name)
                    .squeeze()
                    .drop("variable")
                    .transpose(*da_dims_order)
                )
            else:
                return new_data.to_array(dim=variable_dim, name=da_name).transpose(
                    *da_dims_order
                )
        else:
            return new_data