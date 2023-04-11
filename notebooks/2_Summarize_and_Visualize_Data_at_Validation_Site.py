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

# %%
from dswx_verification.data_models import VerificationParameters
import rasterio
import geopandas as gpd
from matplotlib.colors import ListedColormap
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# %% [markdown]
# # Parameters

# %%
site_name = '3_10'
yaml_file = 'verification_parameters.yml'

# %% [markdown]
# # Load Ids and Set up Directories

# %%
verif_params = VerificationParameters.from_yaml(yaml_file)
verif_params

# %%
all_data_dir = Path(verif_params.data_dir)
all_data_dir.mkdir(exist_ok=True, parents=True)

# %%
site_dir = all_data_dir / site_name
site_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Load Data
#
# ## Rasters

# %%

# %% [markdown]
# ## Metrics

# %% [markdown]
# # Plots at Validation Site

# %%

# %% [markdown]
# # Summarize Metrics at Validation Site

# %%
