import itertools
import numpy as np
import xarray as xr

## Misc ##

def is_dataset_empty(ds:xr.Dataset) -> bool:
    checks_for_empty = [
                        lambda x: len(x.dims) == 0,      # Dataset has no dimensions
                        lambda x: len(x.data_vars) == 0, # Dataset has no variables
                        lambda x: list(x.data_vars.values())[0].count().values == 0 # Data variables are empty
                       ]
    for f in checks_for_empty:
        if f(ds) == True:
            return True
    return False

## End Misc ##

## Combining Data ##

def match_dim_sizes(
    dc, products, x, y, 
    x_y_coords=['longitude', 'latitude'], method='min',
    load_kwargs=None):
    """
    Returns the x and y dimension sizes that match some x and y extents for some products.
    This is useful when determining an absolute resolution to scale products to with
    `xr_scale_res()` in the `aggregate.py` utility file.

    Parameters
    ----------
    dc: datacube.Datacube
        A connection to the Data Cube to determine the resolution of
        individual products from.
    products: list of str
        The names of the products to find a matching resolution for.
    x: list-like
        A list-like of the minimum and maximum x-axis (e.g. longitude) extents for the products.
    y: list-like
        A list-like of the minimum and maximum y-axis (e.g. latitude) extents for the products.
    x_y_coords: list-like or dict
        Either a list-like of the x and y coordinate names or a dictionary mapping product names
        to such list-likes.
    method: str
        The method of finding a matching resolution. The options are
        ['min', 'max'], which separately determine the y and x resolutions
        as the minimum or maximum among all selected products.
    load_kwargs: dict
        Dictionary of product names mapping to parameter dictionaries
        passed to `datacube.Datacube.load()` for the corresponding products.

    Returns
    -------
    abs_res: list
        A list of desired y and x dimension sizes, in that order.
    same_dim_sizes: bool
        Whether all of the dimension sizes were the same.
    """
    coords = []
    if isinstance(x_y_coords, dict):
        for product in products:
            coords.append(x_y_coords[product])
    else:
        coords = [x_y_coords] * len(products)

    load_params = {product:dict(product=product, x=x, y=y, measurements=[]) for product in products}
    if load_kwargs is not None:
        for product in products:
            load_params[product].update(**load_kwargs.get(product, {}) if load_kwargs is not None else {})

    datasets_empty = [dc.load(product=product, measurements=[]) for product in products]

    # First check if all datasets will load with the same x and y dimension sizes.
    same_dim_sizes = True
    first_dataset_dim_size = [datasets_empty[0][coords[0][0]].size, datasets_empty[0][coords[0][1]].size]
    for i in range(1, len(datasets_empty)):
        if first_dataset_dim_size != [datasets_empty[i][coords[i][0]].size, datasets_empty[i][coords[i][1]].size]:
            same_dim_sizes = False
            break

    if method == 'min':
        abs_res = [np.inf, np.inf]
        for i in range(len(datasets_empty)):
            res = [datasets_empty[i][coords[i][0]].size, datasets_empty[i][coords[i][1]].size]
            abs_res[0] = res[0] if res[0] < abs_res[0] else abs_res[0]
            abs_res[1] = res[1] if res[1] < abs_res[1] else abs_res[1]
    else:
        abs_res = [0] * 2
        for i in range(len(datasets_empty)):
            res = [datasets_empty[i][coords[i][0]].size, datasets_empty[i][coords[i][1]].size]
            abs_res[0] = res[0] if abs_res[0] < res[0] else abs_res[0]
            abs_res[1] = res[1] if abs_res[1] < res[1] else abs_res[1]

    return abs_res, same_dim_sizes

## End Combining Data ##

## Extents ##

def get_product_extents(api, platform, product, **kwargs):
    """
    Returns the minimum and maximum latitude, longitude, and date range of a product.

    Parameters
    ----------
    api: DataAccessApi
        An instance of `DataAccessApi` to get query metadata from.
    platform, product: str
        Names of the platform and product to query extent information for.
    **kwargs: dict
        Keyword arguments for `api.get_query_metadata()`.

    Returns
    -------
    full_lat, full_lon: tuple
        Two 2-tuples of the minimum and maximum latitude and longitude, respectively.
    min_max_dates: tuple of datetime.datetime
        A 2-tuple of the minimum and maximum time available.
    """
    # Get the extents of the cube
    descriptor = api.get_query_metadata(platform=platform, product=product, **kwargs)
    min_max_lat = descriptor['lat_extents']
    min_max_lon = descriptor['lon_extents']
    min_max_dates = descriptor['time_extents']
    return min_max_lat, min_max_lon, min_max_dates

def get_overlapping_area(api, platforms, products, **product_kwargs):
    """
    Returns the minimum and maximum latitude, longitude, and date range of the overlapping
    area for a set of products.
    
    Parameters
    ----------
    api: DataAccessApi
        An instance of `DataAccessApi` to get query metadata from.
    platforms, products: list-like of str
        A list-like of names of platforms and products to query extent information for.
        These lists must have the same length.
    **product_kwargs: dict
        A dictionary mapping product names to keyword arguments for
        `get_product_extents()`
        
    Returns
    -------
    full_lat, full_lon: tuple
        Two 2-tuples of the minimum and maximum latitude and longitude, respectively.
    min_max_dates: numpy.ndarray of datetime.datetime
        A 2D NumPy array with shape (len(products), 2), in which rows contain the minimum
        and maximum time available for corresponding products.
    """
    min_max_dates = np.empty((len(platforms), 2), dtype=object)
    min_max_lat = np.empty((len(platforms), 2))
    min_max_lon = np.empty((len(platforms), 2))
    for i, (platform, product) in enumerate(zip(platforms, products)):
        min_max_lat[i], min_max_lon[i], min_max_dates[i] = \
            get_product_extents(api, platform, product,
                                **product_kwargs.get(product, dict()))
    # Determine minimum and maximum lat and lon extents that bound a common area among the
    # products, which are the greatest minimums and smallest maximums.
    min_lon, max_lon = np.max(min_max_lon[:,0]), np.min(min_max_lon[:,1])
    min_lat, max_lat = np.max(min_max_lat[:,0]), np.min(min_max_lat[:,1])
    full_lon = (min_lon, max_lon)
    full_lat = (min_lat, max_lat)
    return full_lat, full_lon, min_max_dates

## End Extents ##