 
from pysteps.visualization.utils import (
    get_geogrid,
    proj4_to_cartopy,
)
from nowproject.utils.plot_precip import _plot_map_cartopy
import cartopy.feature as cfeature

# Only CH
da_event = data_dynamic_ch.sel(time=np.datetime64('2021-06-22T17:40:00.000000000'))["feature"]
METADATA = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 486000.0,
    "y1": 76000.0,
    "x2": 831000.0,
    "y2": 301000.0,
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

# Full rzc
da_event = data_dynamic.sel(time=np.datetime64('2021-06-22T17:40:00.000000000'))["feature"]
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

 


da_event.data = np.ma.masked_invalid(da_event.data)
precip = da_event.data
 
# Assumes the input dimensions are lat/lon
nlat, nlon = precip.shape

x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
    nlat, nlon, geodata=METADATA
)

precip = np.ma.masked_invalid(precip)

# Get colormap and color levels
ptype = "intensity"
units = "mm/h"
colorscale = "pysteps"
cmap, norm, _, _ = get_colormap(ptype, units, colorscale)
crs_ref = proj4_to_cartopy(METADATA["projection"])
crs_proj = crs_ref

### First plot background then array
da_event1 = da_event.copy()
da_event1 = da_event1.assign_coords({"x": da_event1.x.data*1000})
da_event1 = da_event1.assign_coords({"y": da_event1.y.data*1000})

fig, ax = plt.subplots(
            figsize=(8, 5),
            subplot_kw={'projection': crs_proj}
         )
ax = _plot_map_cartopy(crs_proj, 
                       extent=extent,
                       cartopy_scale="50m",
                       drawlonlatlines = True,
                       ax=ax)
ax.set_extent(extent, crs_ref) 

ax.get_zorder() 
ax.get_xlim() #  (255000.0, 965000.0)
ax.get_ylim() # (-160000.0, 480000.0)
ax.get_extent() # (255000.0, 965000.0, -160000.0, 480000.0)     

p = da_event1.plot.imshow(
            ax=ax,
            transform=crs_ref,
            cmap=cmap,
            norm=norm,
            # add_labels=True,
            # extent=extent,
            interpolation="nearest",
            # origin=origin,
            zorder=1,
        )
  
# -----------------------------------------------------.
### Different x and y units m vs km
p = da_event.plot.imshow(
            subplot_kws={'projection': crs_proj},
            transform=crs_ref,
            cmap=cmap,
            norm=norm,
            # add_labels=True,
            # extent=extent,
            interpolation="nearest",
            # origin=origin,
            zorder=1,
        )
ax1 = p.axes
ax1.get_zorder() # zorder has no effect
ax1.get_xlim()   # (255.0, 965.0)
ax1.get_ylim()   # (-160.0, 480.0)
ax1.get_window_extent()     
ax1.get_transform()
ax1.get_extent() # (255.0, 965.0, -160.0, 480.0)


# -----------------------------------------------------.
### Plot multifacetgrid
time_start = np.datetime64('2021-06-22T17:35:00.000000000')
time_end = np.datetime64('2021-06-22T17:42:30.000000000')
# FULL RZC 
da_event = data_dynamic.sel(time=slice(time_start,time_end))["feature"]
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
# RZC CROPPED
da_event = data_dynamic_ch.sel(time=slice(time_start,time_end))["feature"]
METADATA = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 486000.0,
    "y1": 76000.0,
    "x2": 831000.0,
    "y2": 301000.0,
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

# Correct for different coord units
da_event1 = da_event.copy()
da_event1 = da_event1.assign_coords({"x": da_event1.x.data*1000})
da_event1 = da_event1.assign_coords({"y": da_event1.y.data*1000})

# Assumes the input dimensions are lat/lon
# nlat, nlon = precip.shape

# x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
#    nlat, nlon, geodata=METADATA)

# Get colormap and color levels
ptype = "intensity"
units = "mm/h"
colorscale = "pysteps"
cmap, norm, _, _ = get_colormap(ptype, units, colorscale)
crs_ref = proj4_to_cartopy(METADATA["projection"])
crs_proj = crs_ref

p = da_event1.plot.imshow(
            subplot_kws={'projection': crs_proj},
            transform=crs_ref,
            col = "time", col_wrap=2,
            cmap=cmap,
            norm=norm, 
            interpolation="nearest",
            zorder=1,
        )
for ax in p.axes.flatten():
    ax = _plot_map_cartopy(crs_proj, 
                           extent=None, # To be dropped
                           cartopy_scale="50m",
                           drawlonlatlines = True,
                           ax=ax)
    # ax.set_extent(extent, crs_ref)                           
plt.show()  


da_event1 = da_event.copy()
da_event1 = da_event1.assign_coords({"x": da_event1.x.data*1000})
da_event1 = da_event1.assign_coords({"y": da_event1.y.data*1000})

extent = (METADATA["x1"], METADATA["x2"], METADATA["y1"], METADATA["y2"])
fig, ax = plt.subplots(
            figsize=(8, 5),
            subplot_kw={'projection': crs_proj}
         )
p = da_event1.isel(time=0).plot.imshow(
            ax=ax,
            transform=crs_ref,
            cmap=cmap,
            norm=norm, 
            interpolation="nearest",
            zorder=1,
        )

p.axes = _plot_map_cartopy(crs_proj, 
                           extent=None, # To be dropped
                           cartopy_scale="50m",
                           drawlonlatlines = False,
                           ax=p.axes)

ax.set_extent(extent, crs_ref) 




ptype = "intensity"
units = "mm/h"
colorscale = "pysteps"
cmap, norm, _, _ = get_colormap(ptype, units, colorscale)
crs_ref = proj4_to_cartopy(METADATA["projection"])
crs_proj = crs_ref
p = da_event1.plot.imshow(
            subplot_kws={'projection': crs_proj},
            transform=crs_ref,
            col = "time", col_wrap=2,
            cmap=cmap,
            norm=norm, 
            interpolation="nearest",
            zorder=1,
        )

for ax in p.axes.flatten():
    ax = _plot_map_cartopy(crs_proj, 
                           extent=None, # To be dropped
                           cartopy_scale="50m",
                           drawlonlatlines = True,
                           ax=ax)
    ax.set_extent(extent, crs_ref)                           

for i, l_ax in enumerate(p.axes):
    for j, ax in enumerate(l_ax):
        if i != len(p.axes) - 1:
            ax.xaxis.set_ticks([])
        if j != 0:
            ax.yaxis.set_ticks([])

plt.show()  
