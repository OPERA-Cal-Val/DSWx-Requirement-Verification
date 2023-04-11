import warnings

import numpy as np
from dem_stitcher.rio_tools import reproject_arr_to_match_profile


def resample_label_into_percentage(X: np.ndarray,
                                   profile_src: dict,
                                   profile_dst: dict,
                                   class_label: int,
                                   minimum_nodata_percent_for_exclusion: float = .5) -> np.ndarray:
    """Using a given class label, determines percentage of label (excluding np.nans) in new CRS. The output is

    1. Float32 array with each pixel the percentage of label contained in the pixel using resampling `average`
    2. Mask is filled in with np.nan

    The percent calculation *excludes* nodata values. Can determine how to exclude nodata values with
    `minimum_nodata_percent_for_exclusion`.

    Parameters
    ----------
    X : np.ndarray
        Input Labels
    profile_src : dict
        Source profile metadata from rasterio
    profile_dst : dict
        Destination metadata. Relevant keys are transform, dtype (must be float), crs
        and size (width/height)
    class_label : int
        Which class label to resample
    minimum_nodata_percent_for_exclusion : float, optional
        If 1, only pixels with entire nodata will be turned into nodata. Otherwise,
        `mask >= minimum_nodata_percent_for_exclusion` determines slice, by default `.5`.
        A value of 0 (not accepted) means entire image will be masked.

    Returns
    -------
    np.ndarray
        The percent per pixel of the label under consideration with nodata np.nan

    Raises
    ------
    ValueError
        1. If array's dtype is not an integer
        2. If `minimum_nodata_percent_for_exclusion` is > 0 and <= 1
    UserWarning
        If dst dtype is not float
    """
    if ((minimum_nodata_percent_for_exclusion > 1) or
       (minimum_nodata_percent_for_exclusion <= 0)):
        raise ValueError('Minimum_nodata_percent_for_exclusion must be between 0 and 1')

    if X.dtype not in ['uint8', 'int32', 'int64']:
        raise ValueError('Please recast array as integer array')

    if 'float' not in profile_dst['dtype']:
        warnings.warn('dst dtype will be set to float32', category=UserWarning)
    p = profile_src.copy()
    p['dtype'] = 'float32'
    p['nodata'] = np.nan

    X_float = X.astype(np.float32)
    if not np.isnan(profile_src['nodata']):
        nodata = profile_src['nodata']
        X_float[X_float == nodata] = np.nan

    X_true = ((X_float == class_label) & ~np.isnan(X_float)).astype(float)
    X_false = ((X_float != class_label) & ~np.isnan(X_float)).astype(float)
    mask = np.isnan(X_float).astype(float)

    X_true_r, p_perc = reproject_arr_to_match_profile(X_true,
                                                      p,
                                                      profile_dst,
                                                      resampling='average')
    X_true_r = X_true_r[0, ...]
    X_false_r, _ = reproject_arr_to_match_profile(X_false,
                                                  p,
                                                  profile_dst,
                                                  resampling='average')
    X_false_r = X_false_r[0, ...]

    mask_r, _ = reproject_arr_to_match_profile(mask,
                                               p,
                                               profile_dst,
                                               resampling='average')
    mask_r = mask_r[0, ...]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X_percent = X_true_r / (X_true_r + X_false_r)

    X_percent[mask_r >= minimum_nodata_percent_for_exclusion] = np.nan
    return X_percent, p_perc


def reclassify_percentage_array_for_dswx(open_water_percent_array: np.ndarray) -> np.ndarray:
    """Takes an array of where each pixel

    Parameters
    ----------
    open_water_percent_array : np.ndarray
        An array in the DSWx CRS such that each pixel is percentage of open water. Assumes nodata is np.nan

    Returns
    -------
    np.ndarray
        Reclassification

    Raises
    ------
    ValueError
        If min/max water is not between 0/1 respectively; assumes nodata is np.nan
    """
    min_open_water_perc = np.nanmin(open_water_percent_array)
    max_open_water_perc = np.nanmax(open_water_percent_array)
    if (max_open_water_perc > 1) or (min_open_water_perc < 0):
        raise ValueError('The "open_water_percent_array" must be between 0 and 1')

    X_new_labels = np.full(open_water_percent_array.shape, 255, dtype=np.uint8)

    mask = np.isnan(open_water_percent_array)

    ind_w = ~mask & (open_water_percent_array == 1)
    X_new_labels[ind_w] = 1

    ind_pw = ~mask & (open_water_percent_array >= .5) & (open_water_percent_array < 1)
    X_new_labels[ind_pw] = 2

    ind_nw = ~mask & (open_water_percent_array < .5)
    X_new_labels[ind_nw] = 0

    X_new_labels[mask] = 255
    return X_new_labels


def reclassify_validation_dataset_to_dswx_frame(X_val: np.ndarray,
                                                p_val: dict,
                                                p_dswx: dict,
                                                open_water_label: int = 1,
                                                minimum_nodata_percent_for_exclusion: float = .5
                                                ) -> tuple[np.ndarray, dict]:

    nodata_percent = minimum_nodata_percent_for_exclusion
    p_dswx_float = p_dswx.copy()
    p_dswx_float['dtype'] = 'float32'
    X_percent, p_perc = resample_label_into_percentage(X_val,
                                                       p_val,
                                                       p_dswx_float,
                                                       class_label=open_water_label,
                                                       minimum_nodata_percent_for_exclusion=nodata_percent)
    X_val_r = reclassify_percentage_array_for_dswx(X_percent)
    p_val_r = p_perc.copy()
    p_val_r['nodata'] = 255
    p_val_r['dtype'] = np.uint8
    return X_val_r, p_val_r
