# %load_ext autoreload
# %autoreload 2

import torch
from torchvision.models.optical_flow import raft_large, raft_small

import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from nowproject.data.data_utils import (
    prepare_data_dynamic
)

from nowproject.scalers import log_normalize_transform

import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from torchvision.utils import flow_to_image

from nowproject.utils.plot_map import plot_forecast_comparison, plot_forecast_error_comparison
from nowproject.data.data_config import METADATA_CH

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F_vision.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

def transform_data(data, feature_range=(-1, 1)):
    return data * (feature_range[1] - feature_range[0]) + feature_range[0]

def inverse_transform_data(data, threshold: float, feature_min: float, 
                           feature_max: float, feature_range=(-1, 1)):
    data = (data - feature_range[0])/(feature_range[1] - feature_range[0]) 
    data = data * (feature_max - feature_min) + feature_min
    data[data < threshold] = feature_min

    return 10 ** data

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


default_data_dir = "/ltenas3/0_Data/NowProject/"
data_dir_path  = Path(default_data_dir)
boundaries = {"x": slice(485, 831), "y": slice(301, 75)}
data_dynamic = prepare_data_dynamic(data_dir_path / "zarr" / "rzc_temporal_chunk.zarr", 
                                    boundaries=boundaries, timestep=5)


transform_kwargs = dict(feature_min=np.log10(0.025), 
                            feature_max=np.log10(100), 
                            threshold=np.log10(0.1))

timestep_1 = np.datetime64("2017-01-31 16:00:00")
timestep_3 = np.datetime64("2017-01-31 16:10:00")

batch_1 = log_normalize_transform(data_dynamic.sel(time=[timestep_1, timestep_3]).feature, 
                                  **transform_kwargs)
batch_1 = transform_data(np.expand_dims(batch_1.values, 1))
batch_1 = F.pad(torch.Tensor(batch_1), (3, 3, 3, 3), "replicate")
batch_1 = torch.stack([batch_1, batch_1, batch_1], dim=1).reshape(2, 3, 232, 352)

timestep_2 = np.datetime64("2017-01-31 16:05:00")
timestep_4 = np.datetime64("2017-01-31 16:15:00")

batch_2 = log_normalize_transform(data_dynamic.sel(time=[timestep_2, timestep_4]).feature, 
                                  **transform_kwargs)
batch_2 = transform_data(np.expand_dims(batch_2.values, 1))
batch_2 = F.pad(torch.Tensor(batch_2), (3, 3, 3, 3), "replicate")
batch_2 = torch.stack([batch_2, batch_2, batch_2], dim=1).reshape(2, 3, 232, 352)

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

list_of_flows = model(batch_1.to(device), batch_2.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")

predicted_flows = list_of_flows[-1]
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

flow_imgs = flow_to_image(predicted_flows)

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in batch_1]

grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
plot(grid)




out = flow_warp(batch_1.to(device), predicted_flows.permute(0, 2, 3, 1),
                interpolation="bilinear", padding_mode="reflection")

out_rescaled = [(img1 + 1) / 2 for img1 in out]

plt.imshow(out_rescaled[0][0].cpu().detach().numpy())
plt.colorbar()

batch2_rescaled = [(img2 + 1) / 2 for img2 in batch_2]
plt.imshow(out_rescaled[0][0].cpu().detach().numpy() - batch2_rescaled[0][0].cpu().detach().numpy(),
            vmin=-0.6, vmax=0.6)
plt.colorbar()


batch1_rescaled = [(img1 + 1) / 2 for img1 in batch_1]
plt.imshow(batch1_rescaled[0][0].cpu().detach().numpy() - batch2_rescaled[0][0].cpu().detach().numpy(),
            vmin=-0.6, vmax=0.6)
plt.colorbar()

plt.imshow(predicted_flows[0][0].cpu().detach().numpy())
plt.colorbar()





from pysteps.motion.lucaskanade import dense_lucaskanade


timestep_1 = np.datetime64("2017-01-31 16:00:00")
timestep_0 = np.datetime64("2017-01-31 15:55:00")
batch_pysteps = log_normalize_transform(data_dynamic.sel(time=[timestep_0, timestep_1]).feature, 
                                  **transform_kwargs).values
batch_pysteps = F.pad(torch.Tensor(batch_pysteps), (3, 3, 3, 3), "replicate").cpu().detach().numpy()
flow_lk = dense_lucaskanade(batch_pysteps)

out_pysteps = flow_warp(batch_1[0].unsqueeze(0).to(device), torch.Tensor(flow_lk).unsqueeze(0).permute(0, 2, 3, 1).to(device),
                        interpolation="bilinear", padding_mode="reflection")

out_pysteps_rescaled = [(img1 + 1) / 2 for img1 in out_pysteps]

plt.imshow(out_pysteps_rescaled[0][0].cpu().detach().numpy())
plt.colorbar()

plt.imshow(flow_lk[0])
plt.colorbar()

plt.imshow(flow_lk[1])
plt.colorbar()


# --------------------
timestep_1 = np.datetime64("2017-01-31 16:00:00")
timestep_end = np.datetime64("2017-01-31 16:55:00")

batch_1 = log_normalize_transform(data_dynamic.sel(time=slice(timestep_1, timestep_end)).feature, 
                                  **transform_kwargs)
batch_1 = transform_data(np.expand_dims(batch_1.values, 1))
batch_1 = F.pad(torch.Tensor(batch_1), (3, 3, 3, 3), "replicate")
batch_1 = torch.stack([batch_1, batch_1, batch_1], dim=1).reshape(batch_1.shape[0], 3, 232, 352)

timestep_2 = np.datetime64("2017-01-31 16:05:00")
timestep_end_2 = np.datetime64("2017-01-31 17:00:00")

batch_2 = log_normalize_transform(data_dynamic.sel(time=slice(timestep_2, timestep_end_2)).feature, 
                                  **transform_kwargs)
batch_2 = transform_data(np.expand_dims(batch_2.values, 1))
batch_2 = F.pad(torch.Tensor(batch_2), (3, 3, 3, 3), "replicate")
batch_2 = torch.stack([batch_2, batch_2, batch_2], dim=1).reshape(batch_2.shape[0], 3, 232, 352)

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

list_of_flows = model(batch_1.to(device), batch_2.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")

predicted_flows = list_of_flows[-1]

out = flow_warp(batch_1.to(device), predicted_flows.permute(0, 2, 3, 1),
                interpolation="bilinear", padding_mode="reflection")

out = inverse_transform_data(out, **transform_kwargs)
out = out[:, 0, 3:-3, 3:-3].unsqueeze(dim=0).cpu().detach().numpy()

LEADTIME = 12
FREQ_IN_NS = 150000000000 * 2
LEADTIMES = np.arange(FREQ_IN_NS, FREQ_IN_NS*(LEADTIME + 1), FREQ_IN_NS, dtype="timedelta64[ns]")

da = xr.Dataset(
    data_vars=dict(
        feature=(["forecast_reference_time", "leadtime", "y", "x"], out)
    ),
    coords=dict(
        forecast_reference_time=(["forecast_reference_time"], np.array([timestep_1])),
        leadtime=LEADTIMES,
        x=(["x"], data_dynamic.x.data),
        y=(["y"], data_dynamic.y.data),

    )
)

figs_dir = Path("/home/haddad/experiments/optical_flow_figs/")
# plot_forecast_comparison(figs_dir,
#                         da.sel(forecast_reference_time=timestep_1),
#                         data_dynamic,
#                         geodata=METADATA_CH,
#                         suptitle_prefix="Optical Flow RAFT, ")


plot_forecast_error_comparison(figs_dir / "error_figs",
                               da.sel(forecast_reference_time=timestep_1),
                               data_dynamic,
                               geodata=METADATA_CH,
                               suptitle_prefix="Optical Flow RAFT, ")

# --------------------
timestep_1 = np.datetime64("2017-06-14 17:00:00")
timestep_end = np.datetime64("2017-06-14 17:55:00")

batch_1 = log_normalize_transform(data_dynamic.sel(time=slice(timestep_1, timestep_end)).feature, 
                                  **transform_kwargs)
batch_1 = transform_data(np.expand_dims(batch_1.values, 1))
batch_1 = F.pad(torch.Tensor(batch_1), (3, 3, 3, 3), "replicate")
batch_1 = torch.stack([batch_1, batch_1, batch_1], dim=1).reshape(batch_1.shape[0], 3, 232, 352)

timestep_2 = np.datetime64("2017-06-14 17:05:00")
timestep_end_2 = np.datetime64("2017-06-14 18:00:00")

batch_2 = log_normalize_transform(data_dynamic.sel(time=slice(timestep_2, timestep_end_2)).feature, 
                                  **transform_kwargs)
batch_2 = transform_data(np.expand_dims(batch_2.values, 1))
batch_2 = F.pad(torch.Tensor(batch_2), (3, 3, 3, 3), "replicate")
batch_2 = torch.stack([batch_2, batch_2, batch_2], dim=1).reshape(batch_2.shape[0], 3, 232, 352)

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"
# model = raft_large(pretrained=True, progress=False).to(device)
model = raft_small(pretrained=True, progress=False).to(device)
model = model.eval()

list_of_flows = model(batch_1.to(device), batch_2.to(device))
predicted_flows = list_of_flows[-1]

out = flow_warp(batch_1.to(device), predicted_flows.permute(0, 2, 3, 1),
                interpolation="bilinear", padding_mode="reflection")

out = inverse_transform_data(out, **transform_kwargs)
out = out[:, 0, 3:-3, 3:-3].unsqueeze(dim=0).cpu().detach().numpy()

LEADTIME = 12
FREQ_IN_NS = 150000000000 * 2
LEADTIMES = np.arange(FREQ_IN_NS, FREQ_IN_NS*(LEADTIME + 1), FREQ_IN_NS, dtype="timedelta64[ns]")

da = xr.Dataset(
    data_vars=dict(
        feature=(["forecast_reference_time", "leadtime", "y", "x"], out)
    ),
    coords=dict(
        forecast_reference_time=(["forecast_reference_time"], np.array([timestep_1])),
        leadtime=LEADTIMES,
        x=(["x"], data_dynamic.x.data),
        y=(["y"], data_dynamic.y.data),

    )
)

figs_dir = Path("/home/haddad/experiments/optical_flow_figs/")
plot_forecast_comparison(figs_dir,
                        da.sel(forecast_reference_time=timestep_1),
                        data_dynamic,
                        geodata=METADATA_CH,
                        suptitle_prefix="Optical Flow RAFT, ")

diff = da.sel(forecast_reference_time=timestep_1).feature.values - data_dynamic.sel(time=slice(timestep_2, timestep_end_2)).feature.values

plot_forecast_error_comparison(figs_dir / "error_figs/",
                               da.sel(forecast_reference_time=timestep_1),
                               data_dynamic,
                               geodata=METADATA_CH,
                               suptitle_prefix="Optical Flow RAFT, ")


timesteps = [
        # "2016-04-16 18:00:00",
        # "2017-01-12 17:00:00",
        # "2017-01-31 16:00:00",
        # "2017-06-14 16:00:00",
        "2017-07-07 16:00:00",
        "2017-08-31 17:00:00"
    ]

LEADTIME = 12
FREQ_IN_NS = 150000000000 * 2
LEADTIMES = np.arange(FREQ_IN_NS, FREQ_IN_NS*(LEADTIME + 1), FREQ_IN_NS, dtype="timedelta64[ns]")

for timestep in timesteps:
    timestep_1 = np.datetime64(timestep)
    timestep_end = timestep_1 + LEADTIMES[-2]

    batch_1 = log_normalize_transform(data_dynamic.sel(time=slice(timestep_1, timestep_end)).feature, 
                                    **transform_kwargs)
    batch_1 = transform_data(np.expand_dims(batch_1.values, 1))
    batch_1 = F.pad(torch.Tensor(batch_1), (3, 3, 3, 3), "replicate")
    batch_1 = torch.stack([batch_1, batch_1, batch_1], dim=1).reshape(batch_1.shape[0], 3, 232, 352)

    timestep_2 = timestep_1 + LEADTIMES[0]
    timestep_end_2 = timestep_1 + LEADTIMES[-1]

    batch_2 = log_normalize_transform(data_dynamic.sel(time=slice(timestep_2, timestep_end_2)).feature, 
                                    **transform_kwargs)
    batch_2 = transform_data(np.expand_dims(batch_2.values, 1))
    batch_2 = F.pad(torch.Tensor(batch_2), (3, 3, 3, 3), "replicate")
    batch_2 = torch.stack([batch_2, batch_2, batch_2], dim=1).reshape(batch_2.shape[0], 3, 232, 352)

    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()
    list_of_flows = model(batch_1.to(device), batch_2.to(device))
    predicted_flows = list_of_flows[-1]

    out = flow_warp(batch_1.to(device), predicted_flows.permute(0, 2, 3, 1),
                    interpolation="bilinear", padding_mode="reflection")

    out = inverse_transform_data(out, **transform_kwargs)
    out = out[:, 0, 3:-3, 3:-3].unsqueeze(dim=0).cpu().detach().numpy()

    da = xr.Dataset(
        data_vars=dict(
            feature=(["forecast_reference_time", "leadtime", "y", "x"], out)
        ),
        coords=dict(
            forecast_reference_time=(["forecast_reference_time"], np.array([timestep_1])),
            leadtime=LEADTIMES,
            x=(["x"], data_dynamic.x.data),
            y=(["y"], data_dynamic.y.data),

        )
    )

    figs_dir = Path("/home/haddad/experiments/optical_flow_figs/")
    plot_forecast_comparison(figs_dir,
                            da.sel(forecast_reference_time=timestep_1),
                            data_dynamic,
                            geodata=METADATA_CH,
                            suptitle_prefix="Optical Flow RAFT, ")

    diff = da.sel(forecast_reference_time=timestep_1).feature.values - data_dynamic.sel(time=slice(timestep_2, timestep_end_2)).feature.values

    plot_forecast_error_comparison(figs_dir / "error_figs/",
                                da.sel(forecast_reference_time=timestep_1),
                                data_dynamic,
                                geodata=METADATA_CH,
                                suptitle_prefix="Optical Flow RAFT, ")
    

    del model, batch_1, batch_2, out, da, diff, list_of_flows, predicted_flows


#---------------------------------------------------------------------.
## Optical Flow on 1st timestep and apply it to all timesteps

timesteps = [
    "2016-04-16 18:00:00",
    "2017-01-12 17:00:00",
    "2017-01-31 16:00:00",
    "2017-06-14 16:00:00",
    "2017-07-07 16:00:00",
    "2017-08-31 17:00:00"
]

LEADTIME = 12
FREQ_IN_NS = 150000000000 * 2
LEADTIMES = np.arange(FREQ_IN_NS, FREQ_IN_NS*(LEADTIME + 1), FREQ_IN_NS, dtype="timedelta64[ns]")

for timestep in timesteps:
    timestep_1 = np.datetime64(timestep)
    timestep_end = timestep_1 + LEADTIMES[-2]

    batch_1 = log_normalize_transform(data_dynamic.sel(time=[timestep_1]).feature, **transform_kwargs)
    batch_1 = transform_data(np.expand_dims(batch_1.values, 1))
    batch_1 = F.pad(torch.Tensor(batch_1), (3, 3, 3, 3), "replicate")
    batch_1 = torch.stack([batch_1, batch_1, batch_1], dim=1).reshape(batch_1.shape[0], 3, 232, 352)

    timestep_2 = timestep_1 + LEADTIMES[0]
    timestep_end_2 = timestep_1 + LEADTIMES[-1]

    batch_2 = log_normalize_transform(data_dynamic.sel(time=[timestep_2]).feature, **transform_kwargs)
    batch_2 = transform_data(np.expand_dims(batch_2.values, 1))
    batch_2 = F.pad(torch.Tensor(batch_2), (3, 3, 3, 3), "replicate")
    batch_2 = torch.stack([batch_2, batch_2, batch_2], dim=1).reshape(batch_2.shape[0], 3, 232, 352)

    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()
    list_of_flows = model(batch_1.to(device), batch_2.to(device))
    predicted_flows = list_of_flows[-1]

    batch_leadtimes = log_normalize_transform(data_dynamic.sel(time=slice(timestep_1, timestep_end)).feature, **transform_kwargs)
    batch_leadtimes = transform_data(np.expand_dims(batch_leadtimes.values, 1))
    batch_leadtimes = F.pad(torch.Tensor(batch_leadtimes), (3, 3, 3, 3), "replicate")
    batch_leadtimes = torch.stack([batch_leadtimes, batch_leadtimes, batch_leadtimes], dim=1).reshape(batch_leadtimes.shape[0], 3, 232, 352)

    repeated_flow = torch.stack(12*[predicted_flows], dim=0).reshape(12, 2, 232, 352)

    out = flow_warp(batch_leadtimes.to(device), repeated_flow.permute(0, 2, 3, 1),
                    interpolation="bilinear", padding_mode="reflection")

    out = inverse_transform_data(out, **transform_kwargs)
    out = out[:, 0, 3:-3, 3:-3].unsqueeze(dim=0).cpu().detach().numpy()

    da = xr.Dataset(
        data_vars=dict(
            feature=(["forecast_reference_time", "leadtime", "y", "x"], out)
        ),
        coords=dict(
            forecast_reference_time=(["forecast_reference_time"], np.array([timestep_1])),
            leadtime=LEADTIMES,
            x=(["x"], data_dynamic.x.data),
            y=(["y"], data_dynamic.y.data),

        )
    )

    figs_dir = Path("/home/haddad/experiments/optical_flow_figs/fixed_flow_field_all_timesteps/")
    plot_forecast_comparison(figs_dir,
                            da.sel(forecast_reference_time=timestep_1),
                            data_dynamic,
                            geodata=METADATA_CH,
                            suptitle_prefix="Optical Flow RAFT, ")

    diff = da.sel(forecast_reference_time=timestep_1).feature.values - data_dynamic.sel(time=slice(timestep_2, timestep_end_2)).feature.values

    plot_forecast_error_comparison(figs_dir / "error_figs/",
                                da.sel(forecast_reference_time=timestep_1),
                                data_dynamic,
                                geodata=METADATA_CH,
                                suptitle_prefix="Optical Flow RAFT, ")
    

    del model, batch_1, batch_2, out, da, diff, list_of_flows, predicted_flows, batch_leadtimes

#---------------------------------------------------------------------.
## Optical Flow on 1st timestep and apply it 12 times on 1st timestep

timesteps = [
    "2016-04-16 18:00:00",
    "2017-01-12 17:00:00",
    "2017-01-31 16:00:00",
    "2017-06-14 16:00:00",
    "2017-07-07 16:00:00",
    "2017-08-31 17:00:00"
]

LEADTIME = 12
FREQ_IN_NS = 150000000000 * 2
LEADTIMES = np.arange(FREQ_IN_NS, FREQ_IN_NS*(LEADTIME + 1), FREQ_IN_NS, dtype="timedelta64[ns]")

for timestep in timesteps:
    timestep_1 = np.datetime64(timestep)
    timestep_end = timestep_1 + LEADTIMES[-2]

    batch_1 = log_normalize_transform(data_dynamic.sel(time=[timestep_1]).feature, **transform_kwargs)
    batch_1 = transform_data(np.expand_dims(batch_1.values, 1))
    batch_1 = F.pad(torch.Tensor(batch_1), (3, 3, 3, 3), "replicate")
    batch_1 = torch.stack([batch_1, batch_1, batch_1], dim=1).reshape(batch_1.shape[0], 3, 232, 352)

    timestep_2 = timestep_1 + LEADTIMES[0]
    timestep_end_2 = timestep_1 + LEADTIMES[-1]

    batch_2 = log_normalize_transform(data_dynamic.sel(time=[timestep_2]).feature, **transform_kwargs)
    batch_2 = transform_data(np.expand_dims(batch_2.values, 1))
    batch_2 = F.pad(torch.Tensor(batch_2), (3, 3, 3, 3), "replicate")
    batch_2 = torch.stack([batch_2, batch_2, batch_2], dim=1).reshape(batch_2.shape[0], 3, 232, 352)

    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()
    list_of_flows = model(batch_1.to(device), batch_2.to(device))
    predicted_flows = list_of_flows[-1]
    
    input_batch = batch_1
    out = []
    for leadtime in LEADTIMES:
        if len(out) > 0:
            input_batch = out[-1]
        out.append(flow_warp(input_batch.to(device), predicted_flows.permute(0, 2, 3, 1), 
                             interpolation="bilinear", padding_mode="reflection").to(device))
    
    out = torch.stack(out, dim=0).reshape(len(out), 3, 232, 352)
    out = inverse_transform_data(out, **transform_kwargs)
    out = out[:, 0, 3:-3, 3:-3].unsqueeze(dim=0).cpu().detach().numpy()

    da = xr.Dataset(
        data_vars=dict(
            feature=(["forecast_reference_time", "leadtime", "y", "x"], out)
        ),
        coords=dict(
            forecast_reference_time=(["forecast_reference_time"], np.array([timestep_1])),
            leadtime=LEADTIMES,
            x=(["x"], data_dynamic.x.data),
            y=(["y"], data_dynamic.y.data),

        )
    )

    figs_dir = Path("/home/haddad/experiments/optical_flow_figs/fixed_flow_field_only_1st_timestep/")
    # plot_forecast_comparison(figs_dir,
    #                         da.sel(forecast_reference_time=timestep_1),
    #                         data_dynamic,
    #                         geodata=METADATA_CH,
    #                         suptitle_prefix="Optical Flow RAFT, ")

    # diff = da.sel(forecast_reference_time=timestep_1).feature.values - data_dynamic.sel(time=slice(timestep_2, timestep_end_2)).feature.values

    plot_forecast_error_comparison(figs_dir / "error_figs/",
                                da.sel(forecast_reference_time=timestep_1),
                                data_dynamic,
                                geodata=METADATA_CH,
                                suptitle_prefix="Optical Flow RAFT, ")
    

    del model, batch_1, batch_2, out, da, list_of_flows, predicted_flows, input_batch