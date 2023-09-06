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

# %% [markdown]
# # Introduction
#
# The purposes of this notebook is to regenerate the Validation Table that links the following datasets:
#
# 1. Classified Planet Imagery (and their urls)
# 2. DSWx Products (and their urls)
# 3. HLS Products
#
# We also link the so-called "site_name", which is carried over from [Pickens, et al.'s work](https://www.sciencedirect.com/science/article/pii/S0034425720301620) for a obtaining a global stratififed sample of inland water. This notebook puts the table in the `dswx_verification` package so we can read this table using this module as a package.
#
# **Note**: this notebook requires JPL VPN access and a `.env` file created as described in the readme of this repository! Although all the datasets are located in public buckets, the database that makes the searching of these datasets possible is not.

# %%
from dswx_verification import generate_linked_id_table_for_classified_imagery, get_path_of_validation_geojson

# %% [markdown]
# # Generate a new Validation Table

# %%
df = generate_linked_id_table_for_classified_imagery()
df.head()

# %% [markdown]
# # Serialize the Validation Table in package data

# %%
geojson_path = get_path_of_validation_geojson()
geojson_path

# %%
df.to_file(geojson_path, driver='GeoJSON')

# %%
df.to_csv(geojson_path.with_suffix('.csv'), index=False)
