import pathlib
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from typing import Tuple
from nowproject.utils.plot_precip import plot_precip
from pysteps.visualization.utils import proj4_to_cartopy

def plot_obs(figs_dir: pathlib.Path,
             ds_obs: xr.Dataset,
             geodata: dict = None,
             bbox: Tuple[int] = None,
             save_gif: bool = True,
             fps: int = 4,
            ):
    
    figs_dir.mkdir(exist_ok=True)
    (figs_dir / "tmp").mkdir(exist_ok=True)
    # Load in memory
    ds_obs = ds_obs.load()

    var = list(ds_obs.data_vars.keys())[0]
    pil_frames = []

    for i, time in enumerate(ds_obs.time.values):
        time_str = str(time.astype('datetime64[s]'))
        filepath = figs_dir / "tmp" / f"{time_str}.png"
        ##--------------------------------------------------------------------.
        # Create figure
        fig, ax = plt.subplots(
            figsize=(8, 5),
            subplot_kw={'projection': proj4_to_cartopy(geodata["projection"])}
        )
        ##---------------------------------------------------------------------.
        # Plot each variable
        tmp_obs = ds_obs[var].isel(time=i).data
        
        # Plot obs 
        ax = plot_precip(tmp_obs[::-1, :], geodata=geodata, ax=ax)
        ax.set_title("RZC, Time: {}".format(time_str))

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        if save_gif:
            pil_frames.append(Image.open(filepath).convert("P",palette=Image.ADAPTIVE))
 

    if save_gif:
        date = str(pd.to_datetime(ds_obs.time.values[0]).date())
        pil_frames[0].save(
            figs_dir / f"{date}.gif",
            format="gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=1 / fps * 1000,  # ms
            loop=False,
        )