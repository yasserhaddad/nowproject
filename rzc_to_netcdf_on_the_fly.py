import shutil
import pathlib
import datetime

import pyart
import numpy as np
import xarray as xr 

from zipfile import ZipFile
from itertools import repeat
from multiprocessing import Pool

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

def rzc_filename_to_time(filename: str):
    time = datetime.datetime.strptime(filename[3:12], "%y%j%H%M")
    if filename[3:12].endswith("2") or filename[3:12].endswith("7"):
        time = time + datetime.timedelta(seconds=30)

    return time


def read_rzc_file(input_path: pathlib.Path,  
                  row_start: int = 0, row_end: int = 640, 
                  col_start: int = 0, col_end: int = 710) -> xr.Dataset:
    metranet = pyart.aux_io.read_cartesian_metranet(input_path.as_posix())
    rzc = metranet.fields['radar_estimated_rain_rate']['data'][0,:,:]

    x = np.arange(BOTTOM_LEFT_COORDINATES[1], BOTTOM_LEFT_COORDINATES[1] + rzc.shape[1])
    y = np.arange(BOTTOM_LEFT_COORDINATES[0] + rzc.shape[0] - 1, BOTTOM_LEFT_COORDINATES[0] - 1, -1)
    time = rzc_filename_to_time(input_path.as_posix().split("/")[-1])

    ds = xr.Dataset(
                data_vars=dict(
                    precip=(["y", "x"], rzc.data),
                    mask=(["y", "x"], rzc.mask.astype(int)),
                ),
                coords=dict(
                    time=time,
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


def unzip_and_combine_day(input_path: pathlib.Path, output_zip_path: pathlib.Path, 
                          output_netcdf_path: pathlib.Path, log_dir_path: pathlib.Path):
    with ZipFile(input_path, 'r') as zip_ref:
        zip_name = input_path.as_posix().split("/")[-1][:-4]
        output_zip_day_path = output_zip_path / zip_name
        output_zip_day_path.mkdir(exist_ok=True)

        zip_ref.extractall(output_zip_day_path)
        daily_rzc_data_to_netcdf(output_zip_day_path, output_netcdf_path, log_dir_path)
        shutil.rmtree(output_zip_day_path.as_posix())


def unzip_and_combine_rzc(input_dir_path: pathlib.Path, output_zip_path: pathlib.Path,
                          output_netcdf_path: pathlib.Path,  log_dir_path: pathlib.Path,
                          num_workers: int = 2):
    folders = input_dir_path.glob("*")
    for folder in sorted(folders):
        year = int(folder.as_posix().split("/")[-1])
        if year >= 2016:
            print(f"{year}.. ")
            output_zip_year_path = output_zip_path / str(year)
            output_netcdf_year_path = output_netcdf_path / str(year)
            output_zip_year_path.mkdir(exist_ok=True)
            output_netcdf_year_path.mkdir(exist_ok=True)

            list_folders_days = list(sorted(folder.glob("*.zip")))
            with Pool(num_workers) as p:
                p.starmap(unzip_and_combine_day, 
                          zip(list_folders_days, repeat(output_zip_year_path), 
                              repeat(output_netcdf_year_path), repeat(log_dir_path)))



if __name__ == "__main__":
    zip_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/zipped")
    unzipped_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/unzipped_temp")
    netcdf_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/netcdf_temp/")
    log_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/logs_temp/")

    unzipped_dir_path.mkdir(exist_ok=True)
    netcdf_dir_path.mkdir(exist_ok=True)
    log_dir_path.mkdir(exist_ok=True)

    workers = 4
    unzip_and_combine_rzc(zip_dir_path, unzipped_dir_path, netcdf_dir_path, 
                          log_dir_path, num_workers=workers)