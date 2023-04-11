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
                               generate_random_indices_for_classes,
                               get_all_metrics_for_one_trial)
from dswx_verification.data_models import VerificationParameters
from dswx_verification.constants import OSW_ACCURACY_REQ, PSW_ACCURACY_REQ
import yaml
import rasterio
import geopandas as gpd
from matplotlib.colors import ListedColormap
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
from dem_stitcher.rio_window import read_raster_from_window
from itertools import starmap
from tqdm import tqdm
from pandas import json_normalize
import pandas as pd
from pathlib import Path
import json

# %% [markdown]
# # Parameters
#
# We load a parameter file so it can be shared throughout the workflow.

# %%
site_name = '3_10'
yaml_file = 'verification_parameters.yml'

# %% [markdown]
# ## Load parameters
#
# We create a dataclass so that we can easily ensure parameter types are checked.

# %%
verif_params = VerificationParameters.from_yaml(yaml_file)
verif_params

# %% [markdown]
# # Dataset IDs
#
# We get the row of the validation table corresponding to our `site_name`.

# %%
df_site_meta = get_validation_metadata_by_site_name(site_name)
df_site_meta

# %% [markdown]
# Get all the related ids.

# %%
dswx_id = df_site_meta['dswx_id'][0]
planet_id = df_site_meta['planet_id'][0]

# %% [markdown]
# ## Generate directories for site name
#
# These conventions will be shared across the notebooks

# %%
all_data_dir = Path(verif_params.data_dir)
all_data_dir.mkdir(exist_ok=True, parents=True)

# %%
site_dir = all_data_dir / site_name
site_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Load Data
#
# We load the DSWx products and the corresponding validation dataset.
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
#
# This is a quick plot to visualize the products. Subsequent notebooks will finalize plots for presentations.

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
#
# When reprojecting (i.e. resampling or aggregating) the validation data to the DSWx frame we compute the percentage of open water pixels per 30 meter pixels. We are going to save the percent open water raster for later inspection as well.
#
# First, we need to mask out all water areas less than 3 hectares in the validation frame.

# %%
pixels_in_3ha_at_3m = int(3 * get_number_of_pixels_in_hectare(p_val['transform'].a))
size_mask_3m = get_contiguous_areas_of_class_with_maximum_size(X_val, 1, pixels_in_3ha_at_3m)
X_val[size_mask_3m] = 255

# %% [markdown]
# Now, we resample.

# %%
X_perc_r, p_perc_r = resample_label_into_percentage(X_val, p_val, p_dswx_c, 1)


X_val_r, p_val_r = reclassify_validation_dataset_to_dswx_frame(X_val,
                                                               p_val,
                                                               p_dswx_c)

# %% [markdown]
# Again, we plot for interactivity. See subsequent notebooks for finalized plots with colorbars and axes.

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
#
# We are going to perform an additional size mask once the data is reprojected in case their are isolated water bodies that are less than 1 hectare in the DSWx frame.

# %%
pixels_in_3ha_at_30m = int(3 * get_number_of_pixels_in_hectare(p_val_r['transform'].a))
size_mask_30m = get_contiguous_areas_of_class_with_maximum_size(X_val_r, 1, pixels_in_3ha_at_30m)

# %% [markdown]
# ## DSWx Verification Mask
#
# We construct a new shared mask that excludes all pixels that are nodata in the validation or DSWx raster. We also exclude all pixels that are not NW, OSW, PSW (or labels `[0, 1, 2]`) in the DSWx.

# %%
dswx_mask = (size_mask_30m) | (X_val_r == 255) | (~np.isin(X_dswx_c, [0, 1, 2]))

plt.imshow(dswx_mask, interpolation='none')

# %% [markdown]
# # Sampling

# %%
y_val = X_val_r[~dswx_mask]
y_dswx = X_dswx_c[~dswx_mask]

# %% [markdown]
# Based on the validation pixels, we target 1000 total pixels. Using this function, we see the total samples per class. If there aren't enough pixels in a given class, we only sample what is available (without replacement) so we will get less total pixels than what we target.

# %%
samples_per_label = get_equal_samples_per_label(y_val, [0, 1, 2], 1_000)
samples_per_label

# %% [markdown]
# This routine gives us the flattened indices of equal samples from each label with specified number of trials (in this case, this is 1,000).

# %%
sample_indices = generate_random_indices_for_classes(y_val, 
                                                     labels = [0, 1, 2],
                                                     total_target_sample_size=1_000,
                                                     n_trials=100)
y_dswx_trails = [y_dswx[s] for s in sample_indices]
y_val_trials = [y_val[s] for s in sample_indices]

# %% [markdown]
# # Compute Metrics
#
# We are going to collect all the metrics by traversing through all the randomly selected pixels in each trial.

# %%
metrics_for_all_trials = list(starmap(get_all_metrics_for_one_trial, zip(tqdm(y_val_trials), y_dswx_trails)))


# %% [markdown]
# The collected metrics look like this:

# %%
metrics_for_all_trials[0]

# %%
from pandas import json_normalize
df_data_all_trials = pd.DataFrame(json_normalize(metrics_for_all_trials))
df_data_all_trials.head()


# %% [markdown]
# We get the qualitative statistics (i.e. mean and standard deviation) for the various trials.

# %%
df_trials_agg = df_data_all_trials.aggregate(['mean', 'std'])
# includes new column with `*.std` and `*.mean`
df_trials_agg = pd.DataFrame(json_normalize(df_trials_agg.to_dict()))
stat_cols = list(df_trials_agg.columns)

df_trials_agg['dswx_id'] = dswx_id
df_trials_agg['planet_id'] = planet_id
df_trials_agg['site_name'] = site_name
df_trials_agg = df_trials_agg[['site_name', 'planet_id', 'dswx_id'] + stat_cols]
df_trials_agg.head()

# %% [markdown]
# ## Check Requirements

# %%
osw_mean_acc_key = 'acc_per_class.Open_Surface_Water.mean'
mu_osw = df_trials_agg[osw_mean_acc_key].values[0]
psw_mean_acc_key = 'acc_per_class.Partial_Surface_Water.mean'
mu_psw = df_trials_agg[psw_mean_acc_key].values[0]

osw_req_passed = (mu_osw > OSW_ACCURACY_REQ)
df_trials_agg['osw_requirement'] = osw_req_passed

psw_req_passed = (mu_psw > PSW_ACCURACY_REQ)
df_trials_agg['psw_requirement'] = psw_req_passed

# %%
print(f"OSW Requirement Passing: {osw_req_passed} (Acc: {(mu_osw * 100):1.2f}%)")
print(f"PSW Requirement Passing: {psw_req_passed} (Acc: {(mu_psw * 100):1.2f}%)")

# %% [markdown]
# # Serialize
#
# We serialize the rasters we used (for plotting later) and the metrics comparing DSWx and the validation dataset.
#
# ## Rasters

# %%
raster_data_to_serialize = {f'cropped_dswx_{dswx_id}.tif': (X_dswx_c, p_dswx_c, dswx_colormap),
                            f'validation_dataset_{site_name}.tif': (X_val, p_val, dswx_colormap),
                            f'validation_dataset_rprj_{site_name}.tif': (X_val_r, p_val_r, dswx_colormap),
                            f'validation_percent_osw_rprj_{site_name}.tif': (X_perc_r, p_perc_r, None), 
                            f'dswx_mask_{site_name}.tif': (dswx_mask, p_dswx_c, None)}

def write_one(file_name: str, raster: np.ndarray, profile: dict, colormap: dict = None) -> str:
    out_path = site_dir / file_name
    with rasterio.open(out_path, 'w', **profile) as ds:
        ds.write(raster, 1)
        if colormap is not None:
            ds.write_colormap(1, colormap)
        
    return out_path

paths = list(map(lambda item: write_one(item[0], item[1][0], item[1][1], item[1][2]), raster_data_to_serialize.items()))
paths

# %% [markdown]
# ## Metrics

# %%
json_data = df_trials_agg.to_dict('records')[0]

# %%
json.dump(json_data, open(site_dir / 'trial_stats.json', 'w'))
