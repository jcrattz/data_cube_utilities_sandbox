import numpy as np

from .dc_utilities import _get_transform_from_xr
from . import dc_utilities

## Export ##

def export_slice_to_geotiff(ds, path, x_coord='longitude', y_coord='latitude'):
    """
    Exports a single slice of an xarray.Dataset as a GeoTIFF.

    ds: xarray.Dataset
        The Dataset to export. Must have exactly 2 dimensions - 'latitude' and 'longitude'.
    x_coord, y_coord: string
        Names of the x and y coordinates in `ds`.
    path: str
        The path to store the exported GeoTIFF.
    """
    kwargs = dict(tif_path=path, data=ds.astype(np.float32), bands=list(ds.data_vars.keys()),
                  x_coord=x_coord, y_coord=y_coord)
    if 'crs' in ds.attrs:
        kwargs['crs'] = str(ds.attrs['crs'])
    dc_utilities.write_geotiff_from_xr(**kwargs)

## End export ##