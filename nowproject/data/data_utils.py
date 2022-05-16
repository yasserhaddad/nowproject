import pathlib
import xarray as xr

def load_static_topo_data(topo_data_path: pathlib.Path, data_dynamic: xr.Dataset):
    dem = xr.open_rasterio(topo_data_path / "srtm_Switzerland_EPSG21781.tif")
    dem = dem.isel(band=0, drop=True)
    dem = dem.rename({"x": "y", "y": "x"})
    new_y = [y*1000 for y in data_dynamic.y.values[::-1]]
    new_x = [x*1000 for x in data_dynamic.x.values]
    dem = dem.interp(coords={"x": new_x, "y": new_y})
    dem["x"] = dem["x"] / 1000
    dem["y"] = dem["y"] / 1000
    dem = dem.reindex(y=list(reversed(dem.y))).transpose("y", "x")

    return dem