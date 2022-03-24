import zarr
import pyproj

crs = pyproj.CRS.from_epsg(21781)

METADATA = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 255000.0,
    "y1": -160000.0,
    "x2": 965000.0,
    "y2": 480000.0,
    "xpixelsize": 1000.0,
    "ypixelsize": 1000.0,
    "cartesian_unit": "m",
    "yorigin": "upper",
    "institution": "MeteoSwiss",
    "product": "RZC",
    "accutime": 2.5,
    "unit": 'mm/h',
    "zr_a": 316.0,
    "zr_b": 1.5
}

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
    "dtype": "uint16", 
    '_FillValue': 65535,
    "scale_factor": 0.01,
 }

 
mask_zarr_encoding = {
    "dtype": "uint8",
    "_FillValue": 0,
}


ZARR_ENCODING = {
    'precip': precip_zarr_encoding,
    "mask": mask_zarr_encoding
}