import pathlib
import datetime
from zipfile import ZipFile

import pyart
import numpy as np
import pandas as pd
import xarray as xr 

from itertools import repeat
from multiprocessing import Pool, cpu_count

import zarr
import numcodecs
from xforecasting.utils.zarr import write_zarr, rechunk_Dataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)  


BOTTOM_LEFT_COORDINATES = [255, -160]

mask_ntcdf_encoding = {
    'zlib': True,
    'shuffle': True,
    'complevel': 1,
    'fletcher32': False,
    'contiguous': False,
    'chunksizes': (24, 128, 142),
    'dtype': 'uint8'
}

precip_ntcdf_encoding = {
    'zlib': True,
    'shuffle': True,
    'complevel': 1,
    'fletcher32': False,
    'contiguous': False,
    'chunksizes': (24, 128, 142),
    'dtype': "uint16",
    '_FillValue': 65535,
    "scale_factor": 0.01,
    "add_offset": 0.0
 }

NETCDF_ENCODINGS = {
    "mask": mask_ntcdf_encoding,
    "precip": precip_ntcdf_encoding
}


precip_zarr_encoding = {
    "chunks": (25, -1, -1), 
    "compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    "dtype": "int16", 
    '_FillValue': 65535,
    "scale_factor": 0.01,
 }

 
mask_zarr_encoding = {
    "chunks": (25, -1, -1), 
    "compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    "dtype": "uint8",
    "_FillValue": 0,
}


ZARR_ENCODING = {
    'precip': precip_zarr_encoding,
    "mask": mask_zarr_encoding
}


def unzip_files(input_path: pathlib.Path, output_path: pathlib.Path):
    for p in sorted(input_path.glob("*.zip")):
        with ZipFile(p, 'r') as zip_ref:
            zip_name = p.as_posix().split("/")[-1][:-4]
            output_zip_path = output_path / zip_name
            output_zip_path.mkdir(exist_ok=True)
            zip_ref.extractall(output_zip_path)


def unzip_rzc(input_dir_path: pathlib.Path, output_dir_path: pathlib.Path):
    folders = input_dir_path.glob("*")
    for folder in sorted(folders):
        year = int(folder.as_posix().split("/")[-1])
        if year >= 2016:
            print(f"{year}.. ", end="")
            output_year_path = output_dir_path / str(year)
            output_year_path.mkdir(exist_ok=True)
            unzip_files(folder, output_year_path)
            print("done.")
        


def rzc_filename_to_time(filename: str):
    time = datetime.datetime.strptime(filename[3:12], "%y%j%H%M")
    if filename[3:12].endswith("2") or filename[3:12].endswith("7"):
        time = time + datetime.timedelta(seconds=30)

    return time


def read_rzc_file(input_path: pathlib.Path,  
                  row_start: int = 0, row_end: int = 640, 
                  col_start: int = 0, col_end: int = 710) -> xr.Dataset:
    metranet = pyart.aux_io.read_cartesian_metranet(input_path.as_posix(), reader="python")
    rzc = metranet.fields['radar_estimated_rain_rate']['data'][0,:,:]

    x = np.arange(BOTTOM_LEFT_COORDINATES[1], BOTTOM_LEFT_COORDINATES[1] + rzc.shape[1])
    y = np.arange(BOTTOM_LEFT_COORDINATES[0] + rzc.shape[0] - 1, BOTTOM_LEFT_COORDINATES[0] - 1, -1)
    time = rzc_filename_to_time(input_path.as_posix().split("/")[-1])
    radar_availability = input_path.as_posix().split("/")[-1][12:14]

    ds = xr.Dataset(
                data_vars=dict(
                    precip=(["y", "x"], rzc.data),
                    mask=(["y", "x"], rzc.mask.astype(int)),
                ),
                coords=dict(
                    time=time,
                    radar_availability=radar_availability,
                    x=(["x"], x),
                    y=(["y"], y)
                ),
                attrs={},
            )
    
    ds = ds.isel(x=slice(col_start, col_end+1), y=slice(row_start, row_end+1))

    return ds


def daily_rzc_data_to_netcdf(input_dir_path: pathlib.Path, output_dir_path: pathlib.Path,
                             log_dir_path: pathlib.Path, encoding: dict = NETCDF_ENCODINGS, 
                             row_start: int = 0, row_end: int = 640, col_start: int = 0, 
                             col_end: int = 710):
    filename = input_dir_path.as_posix().split("/")[-1]
    output_filename = filename + ".nc"
    list_ds = []

    if not (output_dir_path / output_filename).exists():
        for file in input_dir_path.glob("*.801"):
            try:
                list_ds.append(read_rzc_file(file, row_start, row_end, col_start, col_end))
            except TypeError:
                with open(log_dir_path / f"type_errors_{filename}.txt", "a+") as f:
                    f.write(f"{file.as_posix()}\n")
            except ValueError:
                with open(log_dir_path / f"value_errors_{filename}.txt", "a+") as f:
                    f.write(f"{file.as_posix()}\n")

        if len(list_ds) > 0:
            xr.concat(list_ds, dim="time")\
              .to_netcdf(output_dir_path / output_filename, encoding=encoding)



def rzc_to_netcdf(data_dir_path: pathlib.Path, output_dir_path: pathlib.Path, 
                  log_dir_path: pathlib.Path, num_workers: int = 6):
    for folder_year in sorted(data_dir_path.glob("*")):
        year = int(folder_year.as_posix().split("/")[-1])
        print(year)
        output_dir_year_path = output_dir_path / str(year)
        output_dir_year_path.mkdir(exist_ok=True)

        list_folders_days = list(sorted(folder_year.glob("*")))
        with Pool(num_workers) as p:
            p.starmap(daily_rzc_data_to_netcdf, 
                      zip(list_folders_days, repeat(output_dir_year_path), repeat(log_dir_path)))


def fill_missing_time(ds: xr.Dataset, range_freq: str = "2min30s"):
    start = ds.time.to_numpy()[0]
    end = ds.time.to_numpy()[-1]

    full_range = pd.date_range(start=pd.to_datetime(start).date(), 
                               end=pd.to_datetime(pd.to_datetime(end).date()) + datetime.timedelta(hours=23, minutes=57, seconds=30), 
                               freq=range_freq).to_numpy()
    
    return ds.reindex({"time": full_range}, fill_value={"precip": 65535, "mask": 0})
    


def netcdf_rzc_to_zarr(data_dir_path: pathlib.Path, output_dir_path: pathlib.Path, encoding: dict = ZARR_ENCODING):
    fpaths = [p.as_posix() for p in list(data_dir_path.glob("RZC*.nc"))]
    ds = xr.open_mfdataset(sorted(fpaths))
    ds = fill_missing_time(ds)

    ds = ds.chunk({"time": 25, "y": -1, "x": -1})
    ds.to_zarr((zarr_dir_path / "rzc_temporal_chunk.zarr").as_posix(), encoding=encoding)

    ds = ds.chunk({"time": -1, "y": 1, "x": 1})
    ds.to_zarr((zarr_dir_path / "rzc_spatial_chunk.zarr").as_posix(), encoding=encoding)

    

if __name__ == "__main__":
    zip_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/zipped")
    unzipped_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/unzipped")
    # netcdf_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/netcdf/")
    zarr_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/zarr/")
    log_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/logs/")

    # unzip_rzc(zip_dir_path, unzipped_dir_path)
    # workers = cpu_count() - 4
    # rzc_to_netcdf(unzipped_dir_path, netcdf_dir_path, log_dir_path, num_workers=workers)
    netcdf_dir_path = pathlib.Path("/ltenas3/monika/data_lte")
    netcdf_rzc_to_zarr(netcdf_dir_path, zarr_dir_path)