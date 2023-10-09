# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
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

# %% editable=true slideshow={"slide_type": ""}
from dswx_verification import (get_validation_metadata_by_site_name, 
                               reclassify_validation_dataset_to_dswx_frame, 
                               resample_label_into_percentage,
                               get_contiguous_areas_of_class_with_maximum_size,
                               get_number_of_pixels_in_hectare,
                               get_equal_samples_per_label,
                               generate_random_indices_for_classes,
                               get_all_metrics_for_one_trial,
                               get_geopandas_features_from_array)
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
from shapely.geometry import box
import json

# %% [markdown]
# # Parameters
#
# We load a parameter file so it can be shared throughout the workflow.

# %% tags=["parameters"]
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
hls_id = df_site_meta['hls_id'][0]

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
# We load the validation dataset and corresponding DSWx product.

# %% [markdown]
# ## Val Dataset
#
# Check if the local database path is found in the YML, and if so, use it.

# %%
val_url = df_site_meta['validation_dataset_url'][0]
if verif_params.rel_dswx_db_dir_path:
    val_url = verif_params.dswx_db_dir_parent  / df_site_meta['rel_local_val_path'][0]
val_url

# %%
with rasterio.open(val_url) as ds:
    X_val = ds.read(1)
    p_val = ds.profile
    val_bounds = list(ds.bounds)

# %% [markdown]
# ## DSWx
#
# Again, if the local database path is found in the YML, use it.

# %%
dswx_url = df_site_meta['dswx_urls'][0].split(' ')[0]
if verif_params.rel_dswx_db_dir_path is not None:
    dswx_url = verif_params.dswx_db_dir_parent / df_site_meta['rel_local_dswx_paths'][0].split(' ')[0]
dswx_url

# %%

with rasterio.open(dswx_url) as ds:
    X_dswx = ds.read(1)
    p_dswx = ds.profile
    dswx_colormap = ds.colormap(1)


# %% [markdown]
# We create a dataframe that is $180$ meters (i.e. $60$ pixels $\times$ 3 meters resolution) buffered around validation and then reproject to the DSWx frame to get the bounds in the DSWx frame. This is slightly circuituous than say using the bounds from the `df_site_meta` directly but gives us more transparent control of the buffer with respect to the validation dataset.

# %%
df_val_bounds = gpd.GeoDataFrame(geometry=[box(*val_bounds).buffer(60)],
                                 crs=p_val['crs'])
df_val_bounds = df_val_bounds.to_crs(p_dswx['crs'])
df_val_bounds

# %%
X_dswx_c, p_dswx_c = read_raster_from_window(dswx_url,
                                             df_val_bounds.total_bounds,
                                             df_val_bounds.crs)

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
plt.imshow(X_val == 255)

# %%
p_dswx_c_float = p_dswx_c.copy()
p_dswx_c_float['dtype'] = 'float32'
X_perc_r, p_perc_r = resample_label_into_percentage(X_val, 
                                                    p_val, 
                                                    p_dswx_c_float, 
                                                    1, 
                                                    minimum_nodata_percent_for_exclusion=.5)


X_val_r, p_val_r = reclassify_validation_dataset_to_dswx_frame(X_val,
                                                               p_val,
                                                               p_dswx_c_float,
                                                               open_water_label=1,
                                                               minimum_nodata_percent_for_exclusion=.5)

# %% [markdown]
# Again, we plot for interactivity. See subsequent notebooks for finalized plots with colorbars and axes.

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=250)

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
#
# **Note**: In this case, we exclude [1, 2] *both* as together these define inundation areas.

# %%
pixels_in_3ha_at_30m = int(3 * get_number_of_pixels_in_hectare(p_val_r['transform'].a))
size_mask_30m = get_contiguous_areas_of_class_with_maximum_size(X_val_r, [1, 2], pixels_in_3ha_at_30m)

# %%
plt.imshow(size_mask_30m, interpolation='none', vmax=1, vmin=0)

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
# ## Serialize a set of samples
#
# For posterity - we select the first trial and use the samples from that to serialize into a vector file. Each sample from the trial is given a label `1, ..., total_samples`.

# %%
trial_id = 0
samples = sample_indices[trial_id]

# %% [markdown]
# Want labels to start at 1.

# %%
idx2dswx = {(k+1): label for (k, label) in enumerate(y_dswx[samples])}
idx2val = {(k+1): label for (k, label) in enumerate(y_val[samples])}

# %% [markdown]
# `X_samples` is a raster in which each pixel sampled is given a unique label.

# %%
X_samples = np.zeros(X_dswx_c.shape)
temp = np.zeros(y_val.shape)

temp[sample_indices[trial_id]] = list(idx2dswx.keys())
X_samples[~dswx_mask] = temp

# %%
features = get_geopandas_features_from_array(# Note 8 bits is not enough for 500 points
                                             X_samples.astype(np.int32), 
                                             transform=p_dswx_c['transform'],
                                             mask=(X_samples==0),
                                             label_name='sample_id'
                                            )
df_samples = gpd.GeoDataFrame.from_features(features, 
                                            crs=p_dswx_c['crs'])
df_samples['val_label'] = df_samples['sample_id'].map(lambda label: idx2dswx[label])
df_samples['dswx_label'] = df_samples['sample_id'].map(lambda label: idx2val[label])
df_samples.head()

# %%
fig, ax = plt.subplots()

plot.show(X_val_r, transform=p_val_r['transform'], ax=ax, cmap=cmap, vmin=0, vmax=255, interpolation='none')
df_samples.plot(ax=ax, color='green')

# %%
df_samples.to_file(site_dir / f'samples__{site_name}')

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
# Want to specify metric refers to all classes if normalization of json did not occur
columns_renamed = [f'{col}.All' if '.' not in col else col for col in df_data_all_trials.columns]
df_data_all_trials.columns = columns_renamed
df_data_all_trials.head()


# %% [markdown]
# We get the qualitative statistics (i.e. mean and standard deviation) for the various trials.

# %%
df_trials_agg = df_data_all_trials.aggregate(['mean', 'std', 'median'])
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
raster_data_to_serialize = {f'dswx__{dswx_id}.tif': (X_dswx, p_dswx, dswx_colormap),
                            f'cropped-dswx__{dswx_id}.tif': (X_dswx_c, p_dswx_c, dswx_colormap),
                            f'validation-dataset__{site_name}.tif': (X_val, p_val, dswx_colormap),
                            f'validation-dataset-rprj__{site_name}.tif': (X_val_r, p_val_r, dswx_colormap),
                            f'validation-percent-osw-rprj__{site_name}.tif': (X_perc_r, p_perc_r, None), 
                            f'dswx-mask__{site_name}.tif': (dswx_mask, p_dswx_c, None)}

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

# %% editable=true slideshow={"slide_type": ""}
json.dump(json_data, open(site_dir / 'trial_stats.json', 'w'), indent=2)
