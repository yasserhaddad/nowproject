import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
# https://torchmetrics.readthedocs.io/en/latest/pages/quickstart.html
# https://torchmetrics.readthedocs.io/en/latest/pages/overview.html
# https://torchmetrics.readthedocs.io/en/latest/pages/implement.html#implement
# https://torchmetrics.readthedocs.io/en/latest/references/functional.html#r2-score-func

## feature last dimension:
# - compute per feature
# - torchmetrics multiouput: uniform/variance weighted average across variables or raw ...


# def metrics(loss, list_metrics, reshape_fun)
#     YPRED YOBS = reshape_fun(YPRED YOBS)
#     YPRED YOBS = reshape_fun(YPRED YOBS)
#     LOSS(pred,obs)
#     metrics(prd,obs)

# ------------------------------------------------------------------------------.
##########################
### Custom loss utils ####
##########################
def reshape_tensors_4_loss(Y_pred, Y_obs, dim_info_dynamic, channels_first=False):
    """Reshape tensors for loss computation as currently expexted by WeightedMSELoss ."""
    # Retrieve tensor dimension names
    ordered_dynamic_variables_ = [
        k for k, v in sorted(dim_info_dynamic.items(), key=lambda item: item[1])
    ]

    order = ["feature", "y", "x"] if channels_first else ["y", "x", "feature"]

    # Retrieve dimensions to flat out (consider it as datapoint for loss computation)
    vars_to_flatten = np.array(ordered_dynamic_variables_)[
        np.isin(ordered_dynamic_variables_, order, invert=True)
    ].tolist()
    # Reshape to (data_points, node, feature)

    Y_pred = (
        Y_pred.rename(*ordered_dynamic_variables_)
        .align_to(..., *order)
        .flatten(vars_to_flatten, "data_points")
        .rename(None)
    )
    Y_obs = (
        Y_obs.rename(*ordered_dynamic_variables_)
        .align_to(..., *order)
        .flatten(vars_to_flatten, "data_points")
        .rename(None)
    )
    return Y_pred, Y_obs


# ------------------------------------------------------------------------------.
########################
### Loss definitions ###
########################
class WeightedMSELoss(nn.MSELoss):
    def __init__(self, reduction="mean", weights=None, masked=False):
        super(WeightedMSELoss, self).__init__(reduction="none")
        if not isinstance(reduction, str) or reduction not in ("mean", "mean_masked", "sum", "none"):
            raise ValueError("{} is not a valid value for reduction".format(reduction))
        self.weighted_mse_reduction = reduction

        if weights is not None:
            self.check_weights(weights)
        self.weights = weights

    def forward(self, pred, label):
        if self.weighted_mse_reduction == "mean_masked":
            mask = label > 0
            pred = torch.where(mask, pred, torch.tensor(0.).to(pred.device)) 
        mse = super(WeightedMSELoss, self).forward(pred, label)
        weights = self.weights
        n_batch, n_y, n_x, n_val = mse.shape
        if weights is None:
            weights = torch.ones((n_y, n_x), dtype=mse.dtype, device=mse.device)
        if (n_y, n_x) != weights.shape:
            raise ValueError(
                "The number of weights does not match the the number of pixels. {} != {}".format(
                    mse.shape, (n_y, n_x)
                )
            )
            
        weights = weights.view(1, n_y, n_x, 1).to(mse.device)
        weighted_mse = mse * weights
        if self.weighted_mse_reduction == "sum":
            return torch.sum(weighted_mse) * len(weights)
        elif self.weighted_mse_reduction == "mean" :
            return torch.sum(weighted_mse) / torch.sum(weights) / n_batch / n_val
        elif self.weighted_mse_reduction == "mean_masked":
            len_mask = torch.sum(mask)
            len_mask = len_mask if len_mask > 0 else 1.0
            return torch.sum(weighted_mse) / len_mask
        else:
            return weighted_mse

    def check_weights(self, weights):
        if not isinstance(weights, torch.Tensor):
            raise TypeError(
                "Weights type is not a torch.Tensor. Got {}".format(type(weights))
            )
        if len(weights.shape) != 1:
            raise ValueError("Weights is a 1D vector. Got {}".format(weights.shape))


# ------------------------------------------------------------------------------.
class FSSLoss(nn.Module):
    def __init__(self, mask_size, cutoff=0.5):
        super(FSSLoss, self).__init__()
        self.mask_size = mask_size
        self.cutoff = cutoff

        self.pool1 = nn.AvgPool2d(kernel_size=(self.mask_size, self.mask_size), stride=(1, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(self.mask_size, self.mask_size), stride=(1, 1))

    def forward(self, pred, label):
        # Soft discretization
        c = 10 # make sigmoid function steep
        label_binary = torch.sigmoid(c * (label - self.cutoff))
        pred_binary = torch.sigmoid(c * (pred - self.cutoff))

        label_density = self.pool1(label_binary)
        pred_density = self.pool2(pred_binary)
        n_density_pixels = float(label_density.shape[1] * label_density.shape[2])

        MSE_n = F.mse_loss(label_density, pred_density)
        O_n_squared_image = torch.mul(label_density, label_density)
        # Flatten result, to make it easier to sum over it.
        O_n_squared_vector = torch.flatten(O_n_squared_image)
        # Calculate sum over all terms.
        O_n_squared_sum = torch.sum(O_n_squared_vector)

        M_n_squared_image = torch.mul(pred_density, pred_density)
        # Flatten result, to make it easier to sum over it.
        M_n_squared_vector = torch.flatten(M_n_squared_image)
        # Calculate sum over all terms.
        M_n_squared_sum = torch.sum(M_n_squared_vector)

        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels
        eps = 1e-7
        
        return MSE_n / (MSE_n_ref + eps)