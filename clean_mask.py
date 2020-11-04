import numpy as np
import xarray as xr

## Landsat ##

def landsat_clean_mask_invalid(dataset):
    """
    Masks out invalid data according to the LANDSAT
    surface reflectance specifications. See this document:
    https://landsat.usgs.gov/sites/default/files/documents/ledaps_product_guide.pdf pages 19-20.

    Parameters
    ----------
    dataset: xarray.Dataset
        An `xarray.Dataset` containing bands such as 'red', 'green', or 'blue'.

    Returns
    -------
    invalid_mask: xarray.DataArray
        An `xarray.DataArray` with the same number and order of coordinates as in `dataset`.
        The `True` values specify what pixels are valid.
    """
    invalid_mask = None
    data_arr_names = [arr_name for arr_name in list(dataset.data_vars)
                      if arr_name not in ['pixel_qa', 'radsat_qa', 'cloud_qa']]
    # Only keep data where all bands are in the valid range.
    for i, data_arr_name in enumerate(data_arr_names):
        invalid_mask_arr = (0 < dataset[data_arr_name]) & (dataset[data_arr_name] < 10000)
        invalid_mask = invalid_mask_arr if i == 0 else (invalid_mask & invalid_mask_arr)
    return invalid_mask


def landsat_qa_clean_mask(dataset, platform, cover_types=['clear', 'water']):
    """
    Returns a clean_mask for `dataset` that masks out various types of terrain cover using the
    Landsat pixel_qa band. Note that Landsat masks specify what to keep, not what to remove.
    This means that using `cover_types=['clear', 'water']` should keep only clear land and water.

    See "pixel_qa band" here: https://landsat.usgs.gov/landsat-surface-reflectance-quality-assessment
    and Section 7 here: https://landsat.usgs.gov/sites/default/files/documents/lasrc_product_guide.pdf.

    Parameters
    ----------
    dataset: xarray.Dataset
        An xarray (usually produced by `datacube.load()`) that contains a `pixel_qa` data
        variable.
    platform: str
        A string denoting the platform to be used. Can be "LANDSAT_5", "LANDSAT_7", or
        "LANDSAT_8".
    cover_types: list
        A list of the cover types to include. Adding a cover type allows it to remain in the masked data.
        Cover types for all Landsat platforms include:
        ['fill', 'clear', 'water', 'shadow', 'snow', 'cloud', 'low_conf_cl', 'med_conf_cl', 'high_conf_cl'].

        'fill' removes "no_data" values, which indicates an absense of data. This value is -9999 for Landsat platforms.
        Generally, don't use 'fill'.
        'clear' allows only clear terrain. 'water' allows only water. 'shadow' allows only cloud shadows.
        'snow' allows only snow. 'cloud' allows only clouds, but note that it often only selects cloud boundaries.
        'low_conf_cl', 'med_conf_cl', and 'high_conf_cl' denote low, medium, and high confidence in cloud coverage.
        'low_conf_cl' is useful on its own for only removing clouds, however, 'clear' is usually better suited for this.
        'med_conf_cl' is useful in combination with 'low_conf_cl' to allow slightly heavier cloud coverage.
        Note that 'med_conf_cl' and 'cloud' are very similar.
        'high_conf_cl' is useful in combination with both 'low_conf_cl' and 'med_conf_cl'.

        For Landsat 8, there are more cover types: ['low_conf_cir', 'high_conf_cir', 'terrain_occ'].
        'low_conf_cir' and 'high_conf_cir' denote low and high confidence in cirrus clouds.
        'terrain_occ' allows only occluded terrain.

    Returns
    -------
    clean_mask: xarray.DataArray
        An xarray DataArray with the same number and order of coordinates as in `dataset`.
    """
    from .dc_mosaic import (ls7_unpack_qa, ls8_unpack_qa, ls5_unpack_qa)
    
    processing_options = {
        "LANDSAT_5": ls5_unpack_qa,
        "LANDSAT_7": ls7_unpack_qa,
        "LANDSAT_8": ls8_unpack_qa
    }

    clean_mask = None
    # Keep all specified cover types (e.g. 'clear', 'water'), so logically or the separate masks.
    for i, cover_type in enumerate(cover_types):
        cover_type_clean_mask = processing_options[platform](dataset.pixel_qa, cover_type)
        clean_mask = cover_type_clean_mask if i == 0 else (clean_mask | cover_type_clean_mask)
    return clean_mask

## End Landsat ##