from ast import Str
from multiprocessing.sharedctypes import Value
import time
from typing import List, Union
from torch import threshold
import xarray as xr
import numpy as np
from xscaler.checks import (
    check_variable_dim,
    check_groupby_dims,
    check_rename_dict,
    _check_is_fitted,
    _check_save_fpath,
    get_xarray_variables,
)

from pysteps.utils import transformation

##----------------------------------------------------------------------------.
class RainScaler:
    """Apply log, treat Nans, and MinMaxScaler aggregating over all dimensions 
    (except variable_dim and groupby_dims)."""

    def __init__(
        self,
        feature_min: Union[int, float] = 0,
        feature_max: Union[int, float] = 1,
        threshold: Union[int, float] = 0.1,
    ):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        # Check feature_min, feature_max
        if not isinstance(feature_min, (int, float)):
            raise TypeError("'feature_min' must be a single number.'")
        if not isinstance(feature_max, (int, float)):
            raise TypeError("'feature_max' must be a single number.'")
        if not isinstance(threshold, (int, float)):
            raise TypeError("'threshold' must be a single number.'")
        ##--------------------------------------------------------------------.
        # Initialize
        self.scaler_class = "RainScaler"

        self.feature_min = feature_min
        self.feature_max = feature_max
        self.scaling = self.feature_max - self.feature_min
        self.threshold = threshold

    ##------------------------------------------------------------------------.
    def transform(self, new_data, variable_dim=None, rename_dict=None):
        """Transform rain data to log scale and then normalize the data"""
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
            var_dims = new_data[var].dims

            values = new_data[var].values
            values = np.log10(values + 0.001)
            values[values < self.threshold] = self.feature_min
            values = values.clip(max=self.feature_max)
            values = (values - self.feature_min) / self.scaling
            values[np.isnan(values)] = 0

            new_data[var] = (var_dims, values)
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
            var_dims = new_data[var].dims

            values = new_data[var].values
            values = values * self.scaling + self.feature_min
            values[values < self.threshold] = self.feature_min
            # values[values == self.feature_min] = np.nan
            values = 10 ** values

            new_data[var] = (var_dims, values)
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

##----------------------------------------------------------------------------.
class RainBinScaler:
    """Apply log, treat Nans, and MinMaxScaler aggregating over all dimensions 
    (except variable_dim and groupby_dims)."""

    def __init__(
        self,
        bins: List[Union[int, float]],
        centres: List[Union[int, float]]
    ):
        ##--------------------------------------------------------------------.
        # Initialize
        # Check bins
        if not isinstance(bins, list):
            raise TypeError("'bins' must be a list.'")
        # Check bins
        if not isinstance(centres, list):
            raise TypeError("'centres' must be a list.'")

        self.scaler_class = "RainBinScaler"
        self.bins = bins
        self.centres = centres
        self.inverse_bins = np.vectorize(lambda x: centres[x] if x < len(centres) else centres[-1], otypes=[float])

    ##------------------------------------------------------------------------.
    def transform(self, new_data, variable_dim=None, rename_dict=None):
        """Transform rain data to log scale and then normalize the data"""
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
            var_dims = new_data[var].dims

            values = new_data[var].values
            values = values.clip(max=(self.bins[-1] - 0.01))
            values = np.digitize(values, bins=self.bins)

            new_data[var] = (var_dims, values)
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
            var_dims = new_data[var].dims

            values = new_data[var].values
            values = np.rint(values).astype(int)
            values = self.inverse_bins(values)

            new_data[var] = (var_dims, values)
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

##----------------------------------------------------------------------------.
class dBScaler:
    """Apply log, treat Nans, and MinMaxScaler aggregating over all dimensions 
    (except variable_dim and groupby_dims)."""

    def __init__(
        self,
        threshold: Union[int, float] = 0.1,
        inverse_threshold: Union[int, float] = -10.0,
        zero_value: Union[int, float] = -15.0
    ):
        ##--------------------------------------------------------------------.
        # Check feature_min, feature_max
        if not isinstance(threshold, (int, float)):
            raise TypeError("'threshold' must be a single number.'")
        if not isinstance(inverse_threshold, (int, float)):
            raise TypeError("'threshold' must be a single number.'")
        if not isinstance(zero_value, (int, float)):
            raise TypeError("'threshold' must be a single number.'")
        ##--------------------------------------------------------------------.
        # Initialize
        self.scaler_class = "dBScaler"

        self.threshold = threshold
        self.inverse_threshold = inverse_threshold
        self.zero_value = zero_value

    ##------------------------------------------------------------------------.
    def transform(self, new_data, variable_dim=None, rename_dict=None):
        """Transform rain data to log scale and then normalize the data"""
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
            var_dims = new_data[var].dims

            values = new_data[var].values
            values = transformation.dB_transform(values, threshold=self.threshold, zerovalue=self.zero_value)[0]
            values[~np.isfinite(values)] = self.zero_value
            new_data[var] = (var_dims, values)
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
            var_dims = new_data[var].dims

            values = new_data[var].values
            values = transformation.dB_transform(values, threshold=self.inverse_threshold, inverse=True)[0]
            new_data[var] = (var_dims, values)
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

##----------------------------------------------------------------------------.
class LogScaler:
    """Apply log scaler"""

    def __init__(
        self,
        epsilon: Union[float, int] = 1e-5,
        round_decimals: bool = True
    ):
        # ds_scaler must not be specified. Use load_scaler(fpath) if you want to load an existing scaler from disk.
        ##--------------------------------------------------------------------.
        # Check epsilon
        if not isinstance(epsilon, (int, float)):
            raise TypeError("'epsilon' must be a single number.'")
        ##--------------------------------------------------------------------.
        # Initialize
        self.scaler_class = "logScaler"
        self.epsilon = epsilon
        self.round_decimals = round_decimals

    ##------------------------------------------------------------------------.
    def transform(self, new_data, variable_dim=None, rename_dict=None):
        """Transform rain data to log scale and then normalize the data"""
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
            var_dims = new_data[var].dims

            values = new_data[var].values
            values = np.log10(1 + values/self.epsilon)
            new_data[var] = (var_dims, values)
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
            var_dims = new_data[var].dims

            values = new_data[var].values
            values = (10**values - 1) * self.epsilon
            if self.round_decimals:
                values = values.round(2)
            new_data[var] = (var_dims, values)
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