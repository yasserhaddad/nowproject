import re
import shutil
import pathlib
import datetime
from typing import Any
from zipfile import ZipFile

import pyart
import numpy as np
import pandas as pd
import xarray as xr 

from itertools import repeat
from multiprocessing import Pool, cpu_count

import zarr
from xforecasting.utils.zarr import rechunk_Dataset, write_zarr
from nowproject.data.dataset.data_config import BOTTOM_LEFT_COORDINATES, NETCDF_ENCODINGS, ZARR_ENCODING, METADATA

import warnings
from warnings import warn
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)  


def unzip_files(input_path: pathlib.Path, output_path: pathlib.Path):
    """Unzip .zip files in input_path and save them in output_path.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to input folder
    output_path : pathlib.Path
        Path to folder where unzipped files will be saved
    """
    for p in sorted(input_path.glob("*.zip")):
        with ZipFile(p, 'r') as zip_ref:
            zip_name = p.as_posix().split("/")[-1][:-4]
            output_zip_path = output_path / zip_name
            output_zip_path.mkdir(exist_ok=True)
            zip_ref.extractall(output_zip_path)


def unzip_rzc(input_dir_path: pathlib.Path, output_dir_path: pathlib.Path):
    """Unzip all .zip files of all the different years.

    Parameters
    ----------
    input_dir_path : pathlib.Path
        Path to input folder
    output_dir_path : pathlib.Path
        Path to folder where unzipped files will be saved
    """
    folders = input_dir_path.glob("*")
    for folder in sorted(folders):
        year = int(folder.as_posix().split("/")[-1])
        if year >= 2016:
            print(f"{year}.. ", end="")
            output_year_path = output_dir_path / str(year)
            output_year_path.mkdir(exist_ok=True)
            unzip_files(folder, output_year_path)
            print("done.")    


def rzc_filename_to_time(filename: str) -> datetime.datetime:
    """Determines the time corresponding to the RZC filename.

    Parameters
    ----------
    filename : str
        Name of the RZC file.

    Returns
    -------
    datetime.datetime
        Time corresponding to the RZC filename
    """
    time = datetime.datetime.strptime(filename[3:12], "%y%j%H%M")
    if filename[3:12].endswith("2") or filename[3:12].endswith("7"):
        time = time + datetime.timedelta(seconds=30)

    return time


def get_metranet_header_dictionary(radar_file: str) -> dict:
    """Extracts the header of the RZC file.

    Parameters
    ----------
    radar_file : str
        Path to the RZC file.

    Returns
    -------
    dict
        Header of the RZC file containing the different metadata
    """
    prd_header = {'row': 0, 'column': 0}
    try:
       with open(radar_file, 'rb') as data_file:
           for t_line in data_file:
               line = t_line.decode("utf-8").strip('\n')
               if line.find('end_header') == -1:
                   data = line.split('=')
                   prd_header[data[0]] = data[1]
               else:
                   break
       return prd_header   
    except OSError as ee:
        warn(str(ee))
        print("Unable to read file '%s'" % radar_file)
        return None


def read_rzc_file(input_path: pathlib.Path) -> xr.Dataset:
    """Reads the RZC file and returns an xarray Dataset containing
    the precipitation estimates with the metadata of the file.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to input RZC file

    Returns
    -------
    xr.Dataset
        Dataset containing the precipitation estimates along with
        the metadata of the file
    """
    metranet = pyart.aux_io.read_cartesian_metranet(input_path.as_posix(), reader="python")
    rzc = metranet.fields['radar_estimated_rain_rate']['data'][0,:,:]

    metranet_header = get_metranet_header_dictionary(input_path.as_posix())
    if metranet_header is None:
        metranet_header = {}

    
    x = np.arange(BOTTOM_LEFT_COORDINATES[0], BOTTOM_LEFT_COORDINATES[0] + rzc.shape[1])    
    y = np.arange(BOTTOM_LEFT_COORDINATES[1] + rzc.shape[0] - 1, BOTTOM_LEFT_COORDINATES[1] - 1, -1)

    # Here we define the pixel centroid coordinate
    x = x + 0.5
    y = y + 0.5
    
    time = rzc_filename_to_time(input_path.as_posix().split("/")[-1])
    radar_availability = input_path.as_posix().split("/")[-1][12:14]

    ds = xr.Dataset(
                data_vars=dict(
                    precip=(["y", "x"], rzc.data),
                    mask=(["y", "x"], rzc.mask.astype(int)),
                ),
                coords=dict(
                    time=time,
                    x=(["x"], x),
                    y=(["y"], y),
                    radar_availability=radar_availability,
                    radar_names=metranet_header.get("radar", ""),
                    radar_quality=metranet_header.get("quality", "")
                ),
                attrs={},
            )

    return ds


def daily_rzc_data_to_netcdf(input_dir_path: pathlib.Path, output_dir_path: pathlib.Path,
                             log_dir_path: pathlib.Path, encoding: dict = NETCDF_ENCODINGS):
    """Read all the RZC files of a certain day, combine them into one xarray Dataset and save
    the result in a netCDF file.

    Parameters
    ----------
    input_dir_path : pathlib.Path
        Path to the input folder
    output_dir_path : pathlib.Path
        Path to the output folder
    log_dir_path : pathlib.Path
        Path to save the logs of the reading process
    encoding : dict, optional
        Encoding of the netCDF file, by default NETCDF_ENCODINGS
    """
    filename = input_dir_path.as_posix().split("/")[-1]
    output_filename = filename + ".nc"
    list_ds = []

    if not (output_dir_path / output_filename).exists():
        for file in sorted(input_dir_path.glob("*.801")):
            try:
                list_ds.append(read_rzc_file(file))
            except TypeError:
                with open(log_dir_path / f"type_errors_{filename}.txt", "a+") as f:
                    f.write(f"{file.as_posix()}\n")
            except ValueError:
                with open(log_dir_path / f"value_errors_{filename}.txt", "a+") as f:
                    f.write(f"{file.as_posix()}\n")

        if len(list_ds) > 0:
            xr.concat(list_ds, dim="time", coords="all")\
              .to_netcdf(output_dir_path / output_filename, encoding=encoding)



def rzc_to_netcdf(data_dir_path: pathlib.Path, output_dir_path: pathlib.Path, 
                  log_dir_path: pathlib.Path, num_workers: int = 6):
    """Process all the RZC files and combine them into daily netCDF files.

    Parameters
    ----------
    data_dir_path : pathlib.Path
        Path to the input data folder
    output_dir_path : pathlib.Path
        Path to the output data folder
    log_dir_path : pathlib.Path
        Path to the folder where to save the logs of the
        reading process
    num_workers : int, optional
        Number of workers to parallelize the reading of RZC files, by default 6
    """
    for folder_year in sorted(data_dir_path.glob("*")):
        year = int(folder_year.as_posix().split("/")[-1])
        print(year)
        output_dir_year_path = output_dir_path / str(year)
        output_dir_year_path.mkdir(exist_ok=True)

        list_folders_days = list(sorted(folder_year.glob("*")))
        with Pool(num_workers) as p:
            p.starmap(daily_rzc_data_to_netcdf, 
                      zip(list_folders_days, repeat(output_dir_year_path), repeat(log_dir_path)))


def fill_missing_time(ds: xr.Dataset, range_freq: str = "2min30s") -> xr.Dataset:
    """Fill missing timesteps with default values in a xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset
    range_freq : str, optional
        Interval between each timestep, by default "2min30s"

    Returns
    -------
    xr.Dataset
        Dataset with a contiguous time range and no missing timesteps
    """
    start = ds.time.to_numpy()[0]
    end = ds.time.to_numpy()[-1]

    full_range = pd.date_range(start=pd.to_datetime(start).date(), 
                               end=pd.to_datetime(pd.to_datetime(end).date()) + datetime.timedelta(hours=23, minutes=57, seconds=30), 
                               freq=range_freq).to_numpy()
    
    return ds.reindex({"time": full_range}, fill_value={"precip": 65535, "mask": 0})
    

def drop_duplicates_timesteps(ds: xr.Dataset):
    """Drop duplicate timesteps in the input Dataset by only keeping the
    data measurements taken with the highest number of available radars.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset

    Returns
    -------
    xr.Dataset
        Processed dataset with no duplicate timesteps
    """
    idx_keep = np.arange(len(ds.time))
    to_remove = []
    _, idx, count = np.unique(ds.time, return_counts=True, return_index=True)
    index_duplicates = [list(range(idx[i], idx[i]+count[i])) for i, _ in enumerate(idx) if count[i] > 1]

    for dup in index_duplicates:
        radar_names = [len(re.sub(r'[^a-zA-Z]', '', str(ds.radar_names[i].values))) for i in dup]
        to_remove.extend([dup[i] for i in range(len(radar_names)) if i != radar_names.index(max(radar_names))])

    idx_keep = [i for i in idx_keep if i not in to_remove]

    return ds.isel(time=idx_keep)


def postprocess_netcdf_file(input_file_path: pathlib.Path, output_dir_year_path: pathlib.Path, 
                            encoding: dict = NETCDF_ENCODINGS):
    """Postprocess a daily netCDF file of precipitation estimates, by dropping duplicate timesteps, 
    filling missing timesteps and converting some coordinates to the right type, and then save the
    postprocessed dataset. 

    Parameters
    ----------
    input_file_path : pathlib.Path
        Input netCDF file path
    output_dir_year_path : pathlib.Path
        Output folder where to save the postprocessed dataset
    encoding : dict, optional
        Encoding of the netCDF file, by default NETCDF_ENCODINGS
    """
    ds = fill_missing_time(drop_duplicates_timesteps(xr.open_dataset(input_file_path).sortby("time")))
    ds['radar_quality'] = ds['radar_quality'].astype(str)
    ds['radar_availability'] = ds['radar_availability'].astype(str)
    ds['radar_names'] = ds['radar_names'].astype(str)

    output_filename = input_file_path.as_posix().split("/")[-1]
    ds.to_netcdf(output_dir_year_path / output_filename, encoding=encoding)


def postprocess_all_netcdf(data_dir_path: pathlib.Path, output_dir_path: pathlib.Path, num_workers: int = 6):
    """Postprocess all the daily netCDF files of precipitation estimates, by dropping duplicate timesteps, 
    filling missing timesteps and converting some coordinates to the right type, and then save the
    postprocessed dataset.

    Parameters
    ----------
    data_dir_path : pathlib.Path
        Input folder where the netCDF files are stored
    output_dir_path : pathlib.Path
        Output folder where to save the postprocessed datasets
    num_workers : int, optional
        Number of workers to parallelize the postprocessing of RZC files, by default 6
    """
    for year_dir_path in data_dir_path.glob("*"):
        year = int(year_dir_path.as_posix().split("/")[-1])
        print(year)
        output_dir_year_path = output_dir_path / str(year)
        output_dir_year_path.mkdir(exist_ok=True)
        fpaths = list(year_dir_path.glob("*"))

        with Pool(num_workers) as p:
            p.starmap(postprocess_netcdf_file, zip(fpaths, repeat(output_dir_year_path)))


def netcdf_rzc_to_zarr(data_dir_path: pathlib.Path, output_dir_path: pathlib.Path, 
                       compressor: Any = "auto", encoding: dict = ZARR_ENCODING):
    """Combine all the daily netCDF file into one dataset and save it in Zarr format.

    Parameters
    ----------
    data_dir_path : pathlib.Path
        Input folder where the netCDF files are stored
    output_dir_path : pathlib.Path
        Output folder where to save the dataset in Zarr format
    compressor : Any, optional
        Compressor to pass to the Zarr encoder, by default "auto"
    encoding : dict, optional
        Encoding of the Zarr output, by default ZARR_ENCODING
    """
    temporal_chunk_filepath = output_dir_path / "rzc_temporal_chunk.zarr"
    if temporal_chunk_filepath.exists():
        shutil.rmtree(temporal_chunk_filepath)

    fpaths = [p.as_posix() for p in sorted(list(data_dir_path.glob("*/RZC*.nc")))]
    list_ds = [xr.open_dataset(p, chunks={"time": 576, "x": -1, "y": -1}) for p in fpaths]
    ds = xr.concat(list_ds, dim="time")
    ds.attrs = METADATA

    write_zarr(
        temporal_chunk_filepath.as_posix(),
        ds,
        chunks={"time": 25, "y": -1, "x": -1},
        compressor=compressor,
        rounding=None,
        encoding=encoding,
        consolidated=True,
        append=False,
        show_progress=True,
    )

    ds = xr.open_zarr(temporal_chunk_filepath)
    ds['radar_quality'] = ds['radar_quality'].astype(str)
    ds['radar_availability'] = ds['radar_availability'].astype(str)
    ds['radar_names'] = ds['radar_names'].astype(str)
    
    spatial_chunk_filepath = output_dir_path / "rzc_spatial_chunk.zarr"
    spatial_chunk_temp_filepath = output_dir_path / "rzc_spatial_chunk_temp.zarr"

    if spatial_chunk_filepath.exists():
        shutil.rmtree(spatial_chunk_filepath)
    if spatial_chunk_temp_filepath.exists():
        shutil.rmtree(spatial_chunk_temp_filepath)

    rechunk_Dataset(ds, {"time": -1, "y": 1, "x": 1},
                    spatial_chunk_filepath.as_posix(), 
                    spatial_chunk_temp_filepath.as_posix(), 
                    max_mem="1GB", force=False)
    

if __name__ == "__main__":
    zip_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/zipped")
    unzipped_dir_path = pathlib.Path("/ltenas3/0_Data/NowProject/unzipped")
    netcdf_dir_path = pathlib.Path("/ltenas3/monika/data_lte/rzc/")
    postprocessed_netcdf_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/netcdf/")
    zarr_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/zarr/")
    log_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/logs/")

    workers = cpu_count() - 4
    unzip_rzc(zip_dir_path, unzipped_dir_path)
    # rzc_to_netcdf(unzipped_dir_path, netcdf_dir_path, log_dir_path, num_workers=workers) 
    # postprocess_all_netcdf(netcdf_dir_path, postprocessed_netcdf_dir_path, num_workers=workers)
    # netcdf_rzc_to_zarr(postprocessed_netcdf_dir_path, zarr_dir_path, compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2))