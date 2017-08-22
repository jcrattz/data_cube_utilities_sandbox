# Copyright 2016 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. All Rights Reserved.
#
# Portion of this code is Copyright Geoscience Australia, Licensed under the
# Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License
# at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# The CEOS 2 platform is licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import gdal, osr
import collections
import gc
import numpy as np
import xarray as xr
from datetime import datetime
import collections
from collections import OrderedDict
import hdmedians as hd

import datacube
from . import dc_utilities as utilities


def create_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
    Description:
      Creates a most recent - oldest mosaic of the input dataset. If no clean mask is given,
      the 'cf_mask' variable must be included in the input dataset, as it will be used
      to create a clean mask
    -----
    Inputs:
      dataset_in (xarray.Dataset) - dataset retrieved from the Data Cube; should contain
        coordinates: time, latitude, longitude
        variables: variables to be mosaicked
        If user does not provide a clean_mask, dataset_in must also include the cf_mask
        variable
    Optional Inputs:
      clean_mask (nd numpy array with dtype boolean) - true for values user considers clean;
        if user does not provide a clean mask, one will be created using cfmask
      no_data (int/float) - no data pixel value; default: -9999
    Output:
      dataset_out (xarray.Dataset) - mosaicked data with
        coordinates: latitude, longitude
        variables: same as dataset_in
    """

    dataset_in = dataset_in.copy(deep=True)

    assert clean_mask is not None, "Please provide a boolean clean mask."

    #masks data with clean_mask. all values that are clean_mask==False are set to nodata.
    for key in list(dataset_in.data_vars):
        dataset_in[key].values[np.invert(clean_mask)] = no_data
    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None
    time_slices = reversed(range(len(dataset_in.time))) if kwargs and kwargs['reverse_time'] else range(
        len(dataset_in.time))
    for index in time_slices:
        dataset_slice = dataset_in.isel(time=index).drop('time')
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_in.data_vars):
                dataset_out[key].values[dataset_out[key].values == -9999] = dataset_slice[key].values[dataset_out[key]
                                                                                                      .values == -9999]
                dataset_out[key].attrs = OrderedDict()
    return dataset_out


def create_mean_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
	Description:
		Method for calculating the mean pixel value for a given dataset.
	-----
	Input:
		dataset_in (xarray dataset) - the set of data with clouds and no data removed.
	Optional Inputs:
		no_data (int/float) - no data value.
	"""
    assert clean_mask is not None, "A boolean mask for clean_mask must be supplied."

    dataset_in_filtered = dataset_in.where((dataset_in != no_data) & (clean_mask))
    dataset_out = dataset_in_filtered.mean(dim='time', skipna=True, keep_attrs=False)
    utilities.nan_to_num(dataset_out, no_data)
    #manually clear out dates/timestamps/sats.. median won't produce meaningful reslts for these.
    for key in ['timestamp', 'date', 'satellite']:
        if key in dataset_out:
            dataset_out[key].values[::] = no_data
    return dataset_out.astype(kwargs.get('dtype', 'int32'))


def create_median_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
	Description:
		Method for calculating the median pixel value for a given dataset.
	-----
	Input:
		dataset_in (xarray dataset) - the set of data with clouds and no data removed.
	Optional Inputs:
		no_data (int/float) - no data value.
	"""
    assert clean_mask is not None, "A boolean mask for clean_mask must be supplied."

    dataset_in_filtered = dataset_in.where((dataset_in != no_data) & (clean_mask))
    dataset_out = dataset_in_filtered.median(dim='time', skipna=True, keep_attrs=False)
    utilities.nan_to_num(dataset_out, no_data)
    #manually clear out dates/timestamps/sats.. median won't produce meaningful reslts for these.
    for key in ['timestamp', 'date', 'satellite']:
        if key in dataset_out:
            dataset_out[key].values[::] = no_data
    return dataset_out.astype(kwargs.get('dtype', 'int32'))


def create_max_ndvi_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
	Description:
		Method for calculating the pixel value for the max ndvi value.
	-----
	Input:
		dataset_in (xarray dataset) - the set of data with clouds and no data removed.
	Optional Inputs:
		no_data (int/float) - no data value.
	"""

    dataset_in = dataset_in.copy(deep=True)

    assert clean_mask is not None, "Please provide a boolean clean mask."

    for key in list(dataset_in.data_vars):
        dataset_in[key].values[np.invert(clean_mask)] = no_data

    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None

    time_slices = range(len(dataset_in.time))
    for timeslice in time_slices:
        dataset_slice = dataset_in.isel(time=timeslice).drop('time')
        ndvi = (dataset_slice.nir - dataset_slice.red) / (dataset_slice.nir + dataset_slice.red)
        ndvi.values[np.invert(clean_mask)[timeslice, ::]] = -1000000000
        dataset_slice['ndvi'] = ndvi
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_slice.data_vars):
                dataset_out[key].values[dataset_slice.ndvi.values >
                                        dataset_out.ndvi.values] = dataset_slice[key].values[dataset_slice.ndvi.values >
                                                                                             dataset_out.ndvi.values]
    return dataset_out


def create_min_ndvi_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
	Description:
		Method for calculating the pixel value for the min ndvi value.
	-----
	Input:
		dataset_in (xarray dataset) - the set of data with clouds and no data removed.
	Optional Inputs:
		no_data (int/float) - no data value.
	"""

    dataset_in = dataset_in.copy(deep=True)

    assert clean_mask is not None, "Please provide a boolean clean mask."

    for key in list(dataset_in.data_vars):
        dataset_in[key].values[np.invert(clean_mask)] = no_data

    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None

    time_slices = range(len(dataset_in.time))
    for timeslice in time_slices:
        dataset_slice = dataset_in.isel(time=timeslice).drop('time')
        ndvi = (dataset_slice.nir - dataset_slice.red) / (dataset_slice.nir + dataset_slice.red)
        ndvi.values[np.invert(clean_mask)[timeslice, ::]] = 1000000000
        dataset_slice['ndvi'] = ndvi
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_slice.data_vars):
                dataset_out[key].values[dataset_slice.ndvi.values <
                                        dataset_out.ndvi.values] = dataset_slice[key].values[dataset_slice.ndvi.values <
                                                                                             dataset_out.ndvi.values]
    return dataset_out


def create_geo_median_single_band_mosaic(dataset_in,
                                         clean_mask=None,
                                         no_data=-9999,
                                         intermediate_product=None,
                                         **kwargs):
    """
    Description:
    Calculates the geometric median using single band processing.
    -----
    Input:
    dataset_in (xarray dataset) - the set of data with clouds and no data removed.
    Optional Inputs:
    no_data (int/float) - no data value.
    """
    assert clean_mask is not None, "A boolean mask for clean_mask must be supplied."

    dataset_in_filtered = dataset_in.where((dataset_in != no_data) & (clean_mask))

    output_dict = {}  # Dictionary that will be built out and converted into dataset.
    for band in dataset_in_filtered.keys():
        if band != 'time' and band != 'latitude' and band != 'longitude':
            shape_0, shape_1, shape_2 = dataset_in_filtered[band].values.shape[0], dataset_in_filtered[
                band].values.shape[1], dataset_in_filtered[band].values.shape[2]
            band_data = dataset_in_filtered[band].values.astype(
                'float64')  # Convert to float for purposes of doing algorithm.
            reshaped_data = band_data.reshape((shape_0, shape_1 * shape_2))
            mean_data = np.array(hd.nangeomedian(reshaped_data, axis=0))  # Run the geometric median.
            output_data = mean_data.reshape(shape_1, shape_2)
            output_dict[band] = (('latitude', 'longitude'), output_data)

    dataset_out = xr.Dataset(
        output_dict,
        coords={'latitude': dataset_in_filtered['latitude'],
                'longitude': dataset_in_filtered['longitude']})

    return dataset_out


def create_geo_median_multiple_band_mosaic(dataset_in,
                                           clean_mask=None,
                                           no_data=-9999,
                                           intermediate_product=None,
                                           **kwargs):
    """
    Description:
    Calculates the geometric median using a multi-band processing method.
    -----
    Input:
    dataset_in (xarray dataset) - the set of data with clouds and no data removed.
    Optional Inputs:
    no_data (int/float) - no data value.
    """
    assert clean_mask is not None, "A boolean mask for clean_mask must be supplied."

    dataset_in_filtered = dataset_in.where((dataset_in != no_data) & (clean_mask))

    assert 'blue' in dataset_in_filtered.data_vars, 'No blue band found for the given data.'
    assert 'green' in dataset_in_filtered.data_vars, 'No green band found for the given data.'
    assert 'red' in dataset_in_filtered.data_vars, 'No red band found for the given data.'
    assert 'nir' in dataset_in_filtered.data_vars, 'No nir band found for the given data.'
    assert 'swir1' in dataset_in_filtered.data_vars, 'No swir1 band found for the given data.'
    assert 'swir2' in dataset_in_filtered.data_vars, 'No swir2 band found for the given data.'

    arrays = []
    band_to_index_map = {}
    band_index = 0
    for band in dataset_in_filtered.data_vars:
        if band == 'blue' or band == 'green' or band == 'red' or band == 'nir' or band == 'swir1' or band == 'swir2':
            arrays.append(dataset_in_filtered[band])
            band_to_index_map[band_index] = band
            band_index += 1

    stacked_data = np.stack(arrays)
    bands_shape, time_slices_shape, lat_shape, lon_shape = stacked_data.shape[0], stacked_data.shape[
        1], stacked_data.shape[2], stacked_data.shape[3]

    reshaped_stack = stacked_data.reshape(bands_shape, time_slices_shape,
                                          lat_shape * lon_shape)  # Reshape to remove lat/lon

    vals = np.zeros((bands_shape, lat_shape * lon_shape))  # Build zeroes array across time slices.

    for x in range(reshaped_stack.shape[2]):
        vals[:, x] = hd.nangeomedian(reshaped_stack[:, :, x], axis=1)  # Perform geometric median analysis.

    output_dict = {}

    for key in sorted(band_to_index_map.keys()):
        temp_vals = vals[key, :]
        shaped_sample_output = temp_vals.reshape(lat_shape, lon_shape)
        output_dict[band_to_index_map[key]] = (('latitude', 'longitude'), shaped_sample_output)

    dataset_out = xr.Dataset(
        output_dict, coords={'latitude': dataset_in['latitude'],
                             'longitude': dataset_in['longitude']})

    return dataset_out.astype(kwargs.get('dtype', 'int32'))


def create_medoid_multiple_band_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
    Description:
    Calculates the medoid using a multi-band processing method.
    -----
    Input:
    dataset_in (xarray dataset) - the set of data with clouds and no data removed.
    Optional Inputs:
    no_data (int/float) - no data value.
    """
    assert clean_mask is not None, "A boolean mask for clean_mask must be supplied."

    dataset_in_filtered = dataset_in.where((dataset_in != no_data) & (clean_mask))

    assert 'blue' in dataset_in_filtered.data_vars, 'No blue band found for the given data.'
    assert 'green' in dataset_in_filtered.data_vars, 'No green band found for the given data.'
    assert 'red' in dataset_in_filtered.data_vars, 'No red band found for the given data.'
    assert 'nir' in dataset_in_filtered.data_vars, 'No nir band found for the given data.'
    assert 'swir1' in dataset_in_filtered.data_vars, 'No swir1 band found for the given data.'
    assert 'swir2' in dataset_in_filtered.data_vars, 'No swir2 band found for the given data.'

    arrays = []
    band_to_index_map = {}
    band_index = 0
    for band in dataset_in_filtered.data_vars:
        if band == 'blue' or band == 'green' or band == 'red' or band == 'nir' or band == 'swir1' or band == 'swir2':
            arrays.append(dataset_in_filtered[band])
            band_to_index_map[band_index] = band
            band_index += 1

    stacked_data = np.stack(arrays)
    bands_shape, time_slices_shape, lat_shape, lon_shape = stacked_data.shape[0], stacked_data.shape[
        1], stacked_data.shape[2], stacked_data.shape[3]

    reshaped_stack = stacked_data.reshape(bands_shape, time_slices_shape,
                                          lat_shape * lon_shape)  # Reshape to remove lat/lon
    vals = np.zeros((bands_shape, lat_shape * lon_shape))  # Build zeroes array across time slices.

    for x in range(reshaped_stack.shape[2]):
        vals[:, x] = hd.nanmedoid(reshaped_stack[:, :, x], axis=1)

    output_dict = {}

    for key in sorted(band_to_index_map.keys()):
        temp_vals = vals[key, :]
        shaped_sample_output = temp_vals.reshape(lat_shape, lon_shape)
        output_dict[band_to_index_map[key]] = (('latitude', 'longitude'), shaped_sample_output)

    dataset = xr.Dataset(output_dict, coords={'latitude': dataset_in['latitude'], 'longitude': dataset_in['longitude']})

    return dataset.astype(kwargs.get('dtype', 'int32'))
