 
      
# ------------------------------------------------------------------------.
# from skimage.measure import label, regionprops, regionprops_table
# label_image = label(cleared)
# for region in regionprops(label_image):
#     if region.area >= 100:
#     # draw rectangle around segmented coins
#     minr, minc, maxr, maxc = region.bbox

# def quartiles(regionmask, intensity):
#      return np.percentile(intensity[regionmask], q=(25, 50, 75))
 
# properties = ['label','area', # sum
#               'intensity_min','intensity_mean', 'intensity_max',
#               'centroid', 'centroid_weighted', # centroid_max 
#               'inertia_tensor_eigvals']

# # https://github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/measure/_regionprops.py#L274
# # https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/measure/_regionprops.py#L838-L998

# dict_props = measure.regionprops_table(label_image, intensity=image,
#                                       properties=properties)
#                                      extra_properties=(quartiles,)
 
# The name of the property is derived from the function name, 
# The dtype is inferred by calling the function on a small sample.

# Compile with NUMBA 

     
     
    
 