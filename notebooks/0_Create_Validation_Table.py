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
import datetime
import rasterio
from pathlib import Path
import json
from tqdm import tqdm
from itertools import starmap
import geopandas as gpd
import concurrent.futures
import shutil

import dswx_verification
from dswx_verification import generate_linked_id_table_for_classified_imagery, get_path_of_validation_geojson
from dswx_verification.val_db import get_localized_validation_table, get_classified_planet_table

# %% [markdown]
# # Generate a new Validation Table

# %%
REGENERATE_TABLE_WITH_ES = False

# %%
if not REGENERATE_TABLE_WITH_ES:
    df = get_localized_validation_table()
else:
    df = generate_linked_id_table_for_classified_imagery()
df.head()

# %% [markdown]
# # Localize Data

# %%
LOCALIZE_DATA = True

# %%
t = datetime.date.today()
t

# %%
local_db_dir = Path(f'opera_dswx_val_db-{t.year}{t.month:02d}{t.day:02d}')
local_db_dir.mkdir(exist_ok=True, parents=True)
local_db_dir

# %%
df_planet = get_classified_planet_table()
df_planet.head()


# %%
def download_one(url: str,
                 out_dir: Path,
                 out_file_name: str = None,
                 localize_data=LOCALIZE_DATA) -> Path:
    

    local_file_name = out_file_name or url.split('/')[-1]
    out_path = out_dir / local_file_name

    if localize_data:
        with rasterio.open(url) as ds:
            X, p = ds.read(), ds.profile
    
        with rasterio.open(out_path, 'w', **p) as ds:
            ds.write(X)
    return out_path

def download_many_datasets(urls: list[str],
                           out_dir: Path,
                           max_workers: int = 10) -> list[Path]:
    def download_one_p(url):
        return download_one(url, out_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        out_paths = list(executor.map(download_one_p, urls))
    return out_paths


def download_one_dswx_dataset(urls_str: str,
                              out_dir: Path) -> list[list[Path]]:
    urls = urls_str.split(' ')
    return download_many_datasets(urls, out_dir)

def localize_dswx_data(df: gpd.GeoDataFrame,
                       localize_data=LOCALIZE_DATA) -> list[str]:
    dswx_urls = df.dswx_urls
    site_names = df.site_name

    out_dirs = [local_db_dir / site_name / 'dswx' for site_name in site_names]
    if localize_data:
        [out_dir.mkdir(exist_ok=True, parents=True) for out_dir in out_dirs]
    
    dswx_paths = list(starmap(download_one_dswx_dataset, 
                              zip(tqdm(dswx_urls, desc='DSWx All'), out_dirs)
                             )
                     )
    return dswx_paths

def localize_val_data(df: gpd.GeoDataFrame,
                     localize_data=LOCALIZE_DATA) -> list[str]:
    val_urls = df.validation_dataset_url
    site_names = df.site_name
    planet_ids = df.planet_id

    out_dirs = [local_db_dir / site_name for site_name in site_names]
    if localize_data:
        [out_dir.mkdir(exist_ok=True, parents=True) for out_dir in out_dirs]
    out_file_names = [f'site_name-{sn}-classified_planet-{pid}.tif' for sn, pid in zip(site_names, 
                                                                                       planet_ids)]
    
    val_paths = list(starmap(download_one, 
                             zip(tqdm(val_urls, desc='Val All'), 
                                 out_dirs,
                                 out_file_names)
                             ))
    return val_paths


# %%
dswx_paths = localize_dswx_data(df)

# %%
val_paths = localize_val_data(df)


# %% [markdown]
# ## Serialize Metadata from Planet Classification
#
# This includes all the notes from the manual/semi-automated classification.

# %%
def get_classification_metadata_and_notes(planet_id: str):
    metadata = df_planet[df_planet.image_name == planet_id].to_dict('records')[0]
    return metadata

def serialize_metadata_for_classified_dataset(data: dict):
    site_name = data['site_name']
    planet_id = data['planet_id']
    out_dir = local_db_dir / site_name
    out_path = out_dir / f'Site-{site_name}-metadata.json'
    metadata = get_classification_metadata_and_notes(planet_id)
    # Shapely geometries need to be converted to strings
    metadata['geometry'] = metadata['geometry'].wkt
    json.dump(metadata, open(out_path, 'w'), indent=2)
    return out_path


# %%
records = df.to_dict('records')
metadata_paths = list(map(serialize_metadata_for_classified_dataset, records))

# %% [markdown]
# ## Update Table

# %%
df['rel_local_val_path'] = list(map(str, val_paths))
dswx_paths_str_list = [list(map(str, paths)) for paths in dswx_paths]
df['rel_local_dswx_paths'] = list(map(lambda ps: ' '.join(ps), dswx_paths_str_list))
df.head()

# %% [markdown]
# Save the metadata table inside the local database too.

# %%
df.to_file(local_db_dir / 'validation_table.geojson', driver='GeoJSON')

# %% [markdown]
# We are going to save the `dswx_version` for provenance of the generated data.

# %%
with open(local_db_dir / 'software_version.txt', 'w') as f:
    version=dswx_verification.__version__
    f.write(f'dswx_verification version: {version}')

# %%
if LOCALIZE_DATA:
    shutil.make_archive(local_db_dir, 'zip', local_db_dir)

# %% [markdown]
# # Serialize the Validation Table in package data
#
# This allows us to use the table in the actual package.

# %%
geojson_path = get_path_of_validation_geojson()

# %%
df.to_file(geojson_path, driver='GeoJSON')

# %%
df.to_csv(geojson_path.with_suffix('.csv'), index=False)
