#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:51:57 2022

@author: ghiggi
"""
from warnings import warn
  
def get_metranet_header_dictionary(radar_file):
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
    
fpath = "/home/ghiggi/RZC203221757VL.801"
radar_file = fpath 

header_dict = get_metranet_header_dictionary(fpath)
header_dict["radar"] = 'ADLPW' 
header_dict["quality"] = '77777'
header_dict["data_unit"] = 'mm/h'

#--> To be set as coordinate !!! 

# Global attributes to add to zarr
# - Metadata dictionary compliant with pysteps !

# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/io/importers.py#L27
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/io/nowcast_importers.py#L22
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/io/importers.py#L1159
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/io/importers.py#L1224

metadata = {}

# Define projection 
import pyproj
crs = pyproj.CRS.from_epsg(21781)
metadata['EPSG'] = 21781
metadata['projection'] = crs.to_proj4()
metadata['PROJ_parameters'] = crs.to_json()

metadata["x1"] = 255000.0
metadata["y1"] = -160000.0
metadata["x2"] = 965000.0
metadata["y2"] = 480000.0
metadata["xpixelsize"] = 1000.0
metadata["ypixelsize"] = 1000.0
metadata["cartesian_unit"] = "m"
metadata["yorigin"] = "upper"

metadata["institution"] = "MeteoSwiss"
metadata["product"] = "RZC"
metadata["accutime"] = 2.5
metadata["unit"] = 'mm/h'
metadata["transform"] = None
# metadata["zerovalue"]  
# metadata["threshold"]  
metadata["zr_a"] = 316.0
metadata["zr_b"] = 1.5