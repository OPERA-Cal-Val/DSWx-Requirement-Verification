import os
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from .es_db import get_DSWX_doc

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"


@lru_cache
def get_classified_planet_table() -> gpd.GeoDataFrame:
    """The extents and metadata of the inpendently generated/classified water maps derived from planet data

    Returns
    -------
    gpd.GeoDataFrame
        Table with s3 locations and geometries of classified images (subsets of planet frames)
    """
    df_image_calc = gpd.read_file('s3://opera-calval-database-dswx/image_calc.geojson')

    #  Group by upload date and image name and get the most recent
    df_image_calc = df_image_calc.sort_values(by=['image_name', 'upload_date'], ascending=True)
    df_image_calc = df_image_calc.groupby('image_name').tail(1)

    # Check if "final" is in processing level
    df_image_calc.dropna(subset=['processing_level'], inplace=True)
    temp = df_image_calc.processing_level.str.lower()
    df_image_calc = df_image_calc[temp.str.contains('final')].reset_index(drop=True)
    return df_image_calc


@lru_cache
def get_planet_image_table() -> gpd.GeoDataFrame:
    """Source planet image extents and metadata

    Returns
    -------
    gpd.GeoDataFrame
        Planet imagery with linked HLS id selected by Matthew Bonnema
    """
    return gpd.read_file('s3://opera-calval-database-dswx/image.geojson')


def get_s3_url_of_classified_image(planet_id: str, exclusion_patterns: list = None) -> str:
    """
    Parameters
    ----------
    planet_id : str
    exclusion_patterns : list, optional
        Any string in exclusion pattern if found in s3 key, will be excluded, by default None

    Returns
    -------
    str
       The url of the classified image

    Raises
    ------
    ValueError
        If more than one s3 key is found and no exclusion pattern is specified
    """
    df = get_classified_planet_table()
    df_temp = df[df.image_name == planet_id]
    assert (df_temp.shape[0] == 1)

    record = df_temp.to_dict('records')[0]
    bucket = record['bucket']
    keys = record['s3_keys'].split(',')
    if (len(keys) > 1) and exclusion_patterns:
        keys = list(filter(lambda key: any([patt in key for patt in exclusion_patterns], keys)))

    if len(keys) != 1:
        raise ValueError(f'Specify exclusion patterns to narrow down {len(keys)} s3 keys: {", ".join(keys)}')

    key = keys[0]
    s3_path = f'https://{bucket}.s3.us-west-2.amazonaws.com/{key}'
    return s3_path


def generate_linked_id_table_for_classified_imagery() -> gpd.GeoDataFrame:
    """Links classified datasets with relevant ids needed for validation.

    Returns
    -------
    gpd.GeoDataFrame:
        The geometry is of the classified extents. Links HLS, planet_id, site_name, and DSWx-id. Also obtains the DSWx
        urls
    """
    df_planet = get_planet_image_table()
    df_vd = get_classified_planet_table()
    df_id = pd.merge(df_vd[['image_name', 'geometry', 'water_stratum']],
                     df_planet[['site_name', 'image_name', 'collocated_dswx']],
                     on='image_name')
    df_id = df_id.rename(columns={'image_name': 'planet_id',
                                  'collocated_dswx': 'hls_id'})

    metadata_list = list(map(get_DSWX_doc, tqdm(df_id.hls_id, desc='Retreiving DSWx Metadata')))
    df_id['dswx_id'] = [item['id'] for item in metadata_list]
    product_urls_list = [item['metadata']['product_urls'] for item in metadata_list]
    df_id['dswx_urls'] = [' '.join(urls) for urls in product_urls_list]
    df_id['validation_dataset_url'] = df_id.planet_id.map(get_s3_url_of_classified_image)

    columns = ['site_name', 'planet_id', 'dswx_id', 'hls_id', 'dswx_urls',
               'validation_dataset_url', 'water_stratum', 'geometry']
    df_id = df_id[columns]
    return df_id


def get_path_of_validation_geojson():
    data_dir = Path(__file__).parent.resolve() / 'data'
    return data_dir / 'validation_table.geojson'


def get_localized_validation_table():
    local_geojson_path = get_path_of_validation_geojson()
    return gpd.read_file(local_geojson_path)


def get_validation_metadata_by_site_name(site_name: str) -> pd.DataFrame:
    df_val = get_localized_validation_table()
    df_val_site = df_val[df_val.site_name == site_name].reset_index(drop=True)
    n = df_val_site.shape[0]
    if n != 1:
        raise ValueError('The site name did not yeild a unique row in the localized metadata')
    return df_val_site
