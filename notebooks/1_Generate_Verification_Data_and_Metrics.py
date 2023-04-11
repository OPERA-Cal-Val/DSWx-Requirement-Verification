# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: dswx_val
#     language: python
#     name: dswx_val
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Introduction
#
# This notebook is the core of the validation. It reads validation data and the provisional products, compares the two via random sampling of the classes, and then serializes this information. Although there are plots in this notebook, the visualization, aggregation, and formatting is done in subsequent notebooks.

# %%
from dswx_verification import (get_validation_metadata_by_site_name, 
                               reclassify_validation_dataset_to_dswx_frame, 
                               resample_label_into_percentage,
                               get_contiguous_areas_of_class_with_maximum_size,
                               get_number_of_pixels_in_hectare,
                               get_equal_samples_per_label,
                               generate_random_indices_for_classes)
from dswx_verification.data_models import VerificationParameters
import yaml
import rasterio
import geopandas as gpd
from matplotlib.colors import ListedColormap
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
from dem_stitcher.rio_window import read_raster_from_window

# %% [markdown]
# # Parameters

# %%
site_name = '3_10'
yaml_file = 'verification_parameters.yml'

# %% [markdown]
# ## Load parameters

# %%
verif_params = VerificationParameters.from_yaml(yaml_file)
verif_params

# %% [markdown]
# # Dataset IDs

# %%
df_site_meta = get_validation_metadata_by_site_name(site_name)
df_site_meta

# %% [markdown]
# # Load Data
#
# ## DSWx

# %%
dswx_url = df_site_meta['dswx_urls'][0].split(' ')[0]
with rasterio.open(dswx_url) as ds:
    X_dswx = ds.read(1)
    p_dswx = ds.profile
    dswx_colormap = ds.colormap(1)

# %%
X_dswx_c, p_dswx_c = read_raster_from_window(dswx_url,
                                             df_site_meta.total_bounds,
                                             df_site_meta.crs)

# %% [markdown]
# ## Validation Dataset

# %%
val_url = df_site_meta['validation_dataset_url'][0]
with rasterio.open(val_url) as ds:
    X_val = ds.read(1)
    p_val = ds.profile

# %% [markdown]
# ## Sample Plot

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

cmap = ListedColormap([np.array(dswx_colormap[key]) / 255 for key in range(256)])

plot.show(X_dswx, transform=p_dswx['transform'], ax=ax[0], cmap=cmap, vmin=0, vmax=255, interpolation='none')
df_site_meta.to_crs(p_dswx['crs']).boundary.plot(color='black', ax=ax[0])
ax[0].set_title('Original DSWx Extent with Validation BBox')

plot.show(X_dswx_c, transform=p_dswx_c['transform'], ax=ax[1], cmap=cmap, vmin=0, vmax=255, interpolation='none')
ax[1].set_title('DSWx Frame Cropped')

plot.show(X_val, transform=p_val['transform'], ax=ax[2], cmap=cmap, vmin=0, vmax=255, interpolation='none')
ax[2].set_title('Validation Frame')

# %% [markdown]
# # Reprojection

# %%
pixels_in_3ha_at_3m = int(3 * get_number_of_pixels_in_hectare(p_val['transform'].a))
size_mask_3m = get_contiguous_areas_of_class_with_maximum_size(X_val, 1, pixels_in_3ha_at_3m)
X_val[size_mask_3m] = 255

# %%
X_perc_r, p_perc_r = resample_label_into_percentage(X_val, p_val, p_dswx_c, 1)


X_val_r, p_val_r = reclassify_validation_dataset_to_dswx_frame(X_val,
                                                               p_val,
                                                               p_dswx_c)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))


plot.show(X_val, transform=p_val['transform'], ax=ax[0], cmap=cmap, vmin=0, vmax=255, interpolation='none')
ax[0].set_title('Original Validation Dataset with 3 ha Size Mask')
ax[0].axis('off')

plot.show(X_perc_r, transform=p_val_r['transform'], ax=ax[1], vmin=0, vmax=1, interpolation='none')
ax[1].set_title('Percent OSW in DSWx Frame')
ax[1].axis('off')

plot.show(X_val_r, transform=p_val_r['transform'], ax=ax[2], cmap=cmap, vmin=0, vmax=255, interpolation='none')
ax[2].set_title('Reclassified to DSWx Labels')
ax[2].axis('off')

# %% [markdown]
# # Shared Mask
#
# ## Size Mask

# %%
pixels_in_3ha_at_30m = int(3 * get_number_of_pixels_in_hectare(p_val_r['transform'].a))
size_mask_30m = get_contiguous_areas_of_class_with_maximum_size(X_val_r, 1, pixels_in_3ha_at_30m)

# %% [markdown]
# ## DSWx Verification Mask

# %%
dswx_mask = (size_mask_30m) | (X_val_r == 255) | (~np.isin(X_dswx_c, [0, 1, 2]))

plt.imshow(dswx_mask, interpolation='none')

# %% [markdown]
# # Sampling

# %%
y_val = X_val_r[~dswx_mask]
y_dswx = X_dswx_c[~dswx_mask]

# %%
samples_per_label = get_equal_samples_per_label(y_val, [0, 1, 2], 1_000)
samples_per_label

# %%
sample_indices = generate_random_indices_for_classes(y_val, 
                                                     labels = [0, 1, 2],
                                                     total_target_sample_size=1_000,
                                                     n_trials=100)
y_dswx_samples = np.array([y_dswx[s] for s in sample_indices])
y_val_samples = np.array([y_val[s] for s in sample_indices])

# %% [markdown]
# # Compute Metrics

# %%


# %% [markdown]
# # Serialize

# %%
