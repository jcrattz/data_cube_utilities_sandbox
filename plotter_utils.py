from collections import OrderedDict
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap

from scipy.interpolate import CubicSpline
import time

from .curve_fitting import gaussian_fit, gaussian_filter_fit, poly_fit, fourier_fit
from .scale import xr_scale, np_scale
from .dc_time import _n64_to_datetime, _n64_datetime_to_scalar, _scalar_to_n64_datetime


from .plotter_utils_consts import n_pts_smooth, default_fourier_n_harm

## Datetime functions ##

def n64_to_epoch(timestamp):
    ts = pd.to_datetime(str(timestamp))
    time_format = "%Y-%m-%d"
    ts = ts.strftime(time_format)
    epoch = int(time.mktime(time.strptime(ts, time_format)))
    return epoch


def np_dt64_to_str(np_datetime, fmt='%Y-%m-%d'):
    """Converts a NumPy datetime64 object to a string based on a format string supplied to pandas strftime."""
    return pd.to_datetime(str(np_datetime)).strftime(fmt)


## End datetime functions ##


def xarray_time_series_plot(dataset, plot_descs, x_coord='longitude',
                            y_coord='latitude', fig_params=None,
                            fig=None, ax=None, show_legend=True, title=None,
                            max_times_per_plot=None, max_cols=1):
    """
    Plot data variables in an xarray.Dataset together in one figure, with different
    plot types for each (e.g. box-and-whisker plot, line plot, scatter plot), and
    optional curve fitting to aggregations along time. Handles data binned with
    xarray.Dataset methods resample() and groupby(). That is, it handles data
    binned along time (e.g. by week) or across years (e.g. by week of year).

    Parameters
    -----------
    dataset: xarray.Dataset
        A Dataset containing some bands like NDVI or WOFS.
        It must have time, x, and y coordinates with names specified by
        the 'x_coord' and 'y_coord' parameters.
    plot_descs: dict
        Dictionary mapping names of DataArrays in the Dataset to plot to
        dictionaries mapping aggregation types (e.g. 'mean', 'median') to
        lists of dictionaries mapping plot types
        (e.g. 'line', 'box', 'scatter') to keyword arguments for plotting.

        Aggregation happens within time slices and can be many-to-many or many-to-one.
        Some plot types require many-to-many aggregation (e.g. 'none'), and some other plot types
        require many-to-one aggregation (e.g. 'mean'). Aggregation types can be any of
        ['min', 'mean', 'median', 'none', 'max'], with 'none' performing no aggregation.

        Plot types can be any of
        ['scatter', 'line', 'box', 'gaussian', 'gaussian_filter', 'poly', 'cubic_spline', 'fourier'].
        Here are the required arguments, with format {plot_type: {arg_name: (data_type[, description]}}:
        {'poly':
            {'degree': (int, "the degree of the polynomial to fit.")}}
        Here are the optional arguments, with format {plot_type: {arg_name: (data_type[, description]}}:
        {'box': # See matplotlib.axes.Axes.boxplot() for more information.
            {'boxprops': dict, 'flierprops': dict, 'showfliers': bool},
         'gaussian_filter': # See gaussian_filter_fit() in data_cube_utilities/curve_fitting.py for more information.
             {'sigma': numeric},
         'fourier':
            {'extrap_time': (string, "a positive integer followed by Y, M, or D -
                                      year, month, or day - specifying the
                                      amount of time to extrapolate over."),
             'extrap_color': (matplotlib color, "a matplotlib color to color the extrapolated data with.")}}
        Additionally, all of the curve fits (['gaussian', 'gaussian_filter', 'poly',
        'cubic_spline', 'fourier']) support an optional 'smooth' boolean parameter.
        If true, the curve fit is smoothed, otherwise it will look no smoother than the original data.

        Here is an example:
        {'ndvi':{'mean':[{'line':{'color':'forestgreen', 'alpha':alpha}}],
                 'none':[{'box':{'boxprops':{'facecolor':'forestgreen','alpha':alpha},
                                 'showfliers':False}}]}}
        This example will create a green line plot of the mean of the 'ndvi' band
        as well as a green box plot of the 'ndvi' band.
    x_coord, y_coord: str
        Names of the x and y coordinates in `dataset`.
    fig_params: dict
        Figure parameters dictionary (e.g. {'figsize':(12,6)}). Used to create a Figure
        ``if fig is None and ax is None`.
    fig: matplotlib.figure.Figure
        The figure to use for the plot.
        If only `fig` is supplied, the Axes object used will be the first. This
        argument is ignored if ``max_times_per_plot`` is less than the number of times.
    ax: matplotlib.axes.Axes
        The axes to use for the plot. This argument is ignored if
        ``max_times_per_plot`` is less than the number of times.
    show_legend: bool
        Whether or not to show the legend.
    title: str
        The title of each subplot. Note that a date range enclosed in parenthesis
        will be postpended whether this is specified or not.
    max_times_per_plot: int
        The maximum number of times per plot. If specified, multiple plots may be created,
        with each plot having as close to `num_times/max_times_per_plot` number of points
        as possible, where `num_times` is the total number of plotting points, including
        extrapolations. The plots will be arranged in a row-major grid, with the number
        of columns being at most `max_cols`.
    max_cols: int
        The maximum number of columns in the plot grid.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure containing the plot grid.
    plotting_data: dict
        A dictionary mapping 3-tuples of data array names, aggregation types, and plot types
        (e.g. ('ndvi', 'none', 'box')) to `xarray.DataArray` objects of the data that was
        plotted for those combinations of aggregation types and plot types.

    Raises
    ------
    ValueError:
        If an aggregation type is not possible for a plot type

    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    fig_params = {} if fig_params is None else fig_params

    # Lists of plot types that can and cannot accept many-to-one aggregation
    # for each time slice, as well as plot types that support extrapolation.
    plot_types_requiring_aggregation = ['line', 'gaussian', 'gaussian_filter', 'poly',
                                        'cubic_spline', 'fourier']
    plot_types_handling_aggregation = ['scatter'] + plot_types_requiring_aggregation
    plot_types_not_handling_aggregation = ['box']
    plot_types_curve_fit = ['gaussian', 'gaussian_filter', 'poly',
                            'cubic_spline', 'fourier']
    plot_types_supporting_extrapolation = ['fourier']
    all_plot_types = list(set(plot_types_requiring_aggregation + plot_types_handling_aggregation + \
                              plot_types_not_handling_aggregation + plot_types_curve_fit + \
                              plot_types_supporting_extrapolation))

    # Aggregation types that aggregate all values for a given time to one value.
    many_to_one_agg_types = ['min', 'mean', 'median', 'max']
    # Aggregation types that aggregate to many values or do not aggregate.
    many_to_many_agg_types = ['none']
    all_agg_types = many_to_one_agg_types + many_to_many_agg_types

    # Determine how the data was aggregated, if at all.
    possible_time_agg_strs = ['time', 'week', 'month']
    time_agg_str = 'time'
    for possible_time_agg_str in possible_time_agg_strs:
        if possible_time_agg_str in list(dataset.coords):
            time_agg_str = possible_time_agg_str
            break
    # Make the data 2D - time and a stack of all other dimensions.
    all_plotting_data_arrs = list(plot_descs.keys())
    all_plotting_data = dataset[all_plotting_data_arrs]
    all_times = all_plotting_data[time_agg_str].values
    # Mask out times for which no data variable to plot has any non-NaN data.
    nan_mask_data_vars = list(all_plotting_data[all_plotting_data_arrs] \
                              .notnull().data_vars.values())
    for i, data_var in enumerate(nan_mask_data_vars):
        time_nan_mask = data_var if i == 0 else time_nan_mask | data_var
    time_nan_mask = time_nan_mask.any([x_coord, y_coord])
    times_not_all_nan = all_times[time_nan_mask.values]
    non_nan_plotting_data = all_plotting_data.loc[{time_agg_str: times_not_all_nan}]

    # Determine the number of extrapolation data points. #
    extrap_day_range = 0
    n_extrap_pts = 0
    # For each data array to plot...
    for data_arr_name, agg_dict in plot_descs.items():
        # For each aggregation type (e.g. 'mean', 'median')...
        for agg_type, plot_dicts in agg_dict.items():
            # For each plot for this aggregation type...
            for i, plot_dict in enumerate(plot_dicts):
                for plot_type, plot_kwargs in plot_dict.items():
                    # Only check the plot types supporting extrapolation.
                    if plot_type == 'fourier':
                        curr_extrap_day_range = 0
                        n_predict = 0  # Default to no extrapolation.
                        # Addressing this way to modify `plot_descs`.
                        extrap_time = plot_kwargs.get('extrap_time', None)
                        if extrap_time is not None:
                            assert time_agg_str == 'time', \
                                "Extrapolating for data with a time dimension other than 'time' - " \
                                "such as 'month', or 'week' - is not supported. A time dimension of 'month' " \
                                "or 'week' denotes data aggregated for each month or week across years, so " \
                                "extrapolation is meaningless in that case. Support for a time dimension of 'year' " \
                                "has not yet been added."
                            # Determine the number of points to extrapolate (in an approximate manner).
                            # First find the time range of the given data.
                            first_last_days = list(map(lambda np_dt_64: _n64_to_datetime(np_dt_64),
                                                       non_nan_plotting_data.time.values[[0, -1]]))
                            year_range = first_last_days[1].year - first_last_days[0].year
                            month_range = first_last_days[1].month - first_last_days[0].month
                            day_range = first_last_days[1].day - first_last_days[0].day
                            day_range = year_range * 365.25 + month_range * 30 + day_range
                            # Then find the time range of the extrapolation string.
                            fields = re.match(r"(?P<num>[0-9]{0,5})(?P<unit>[YMD])", extrap_time)
                            assert fields is not None, \
                                r"For the '{}' DataArray: When using 'fourier' as " \
                                "the fit type, if the 'extrap_time' parameter is supplied, it must be " \
                                "a string containing a positive integer followed by one of ['Y', 'M', or 'D']." \
                                    .format(data_arr_name)
                            num, unit = int(fields['num']), fields['unit']
                            days_per_unit = dict(Y=365.25, M=30, D=1)[unit]
                            curr_extrap_day_range = num * days_per_unit
                            n_predict = round(len(non_nan_plotting_data[time_agg_str]) *
                                              (curr_extrap_day_range / day_range))
                            plot_descs[data_arr_name][agg_type][i][plot_type] \
                                ['n_predict'] = n_predict
                        # This parameter is used by get_curvefit() later.
                        extrap_day_range = max(extrap_day_range, curr_extrap_day_range)
                        n_extrap_pts = max(n_extrap_pts, n_predict)

    # Collect (1) the times not containing only NaN values and (2) the extrapolation times.
    if time_agg_str == 'time' and len(times_not_all_nan) > 0:
        first_extrap_time = times_not_all_nan[-1] + np.timedelta64(extrap_day_range, 'D') / n_extrap_pts
        last_extrap_time = times_not_all_nan[-1] + np.timedelta64(extrap_day_range, 'D')
        extrap_times = np.linspace(_n64_datetime_to_scalar(first_extrap_time),
                                   _n64_datetime_to_scalar(last_extrap_time), num=n_extrap_pts)
        extrap_times = np.array(list(map(_scalar_to_n64_datetime, extrap_times)))
        times_not_all_nan_and_extrap = np.concatenate((times_not_all_nan, extrap_times)) \
            if len(extrap_times) > 0 else times_not_all_nan
    else:
        times_not_all_nan_and_extrap = times_not_all_nan
    # Compute all of the plotting data - handling aggregations and extrapolations.
    plotting_data_not_nan_and_extrap = {}  # Maps data arary names to plotting data (NumPy arrays).
    # Get the x locations of data points not filled with NaNs and the x locations of extrapolation points.
    epochs = np.array(list(map(n64_to_epoch, times_not_all_nan_and_extrap))) \
        if time_agg_str == 'time' else times_not_all_nan_and_extrap
    epochs_not_extrap = epochs[:len(times_not_all_nan)]

    # Handle aggregations and curve fits. #
    # For each data array to plot...
    for data_arr_name, agg_dict in plot_descs.items():
        data_arr_plotting_data = non_nan_plotting_data[data_arr_name]
        # For each aggregation type (e.g. 'mean', 'median')...
        for agg_type, plot_dicts in agg_dict.items():
            # For each plot for this aggregation type...
            for i, plot_dict in enumerate(plot_dicts):
                for plot_type, plot_kwargs in plot_dict.items():
                    assert plot_type in all_plot_types, \
                        r"For the '{}' DataArray: plot_type '{}' not recognized" \
                            .format(data_arr_name, plot_type)

                    # Ensure aggregation types are legal for this data.
                    # Some plot types require aggregation.
                    if plot_type in plot_types_requiring_aggregation:
                        if agg_type not in many_to_one_agg_types:
                            raise ValueError("For the '{}' DataArray: the plot type "
                                             "'{}' only accepts many-to-one aggregation (currently using '{}'). "
                                             "Please pass any of {} as the aggregation type "
                                             "or change the plot type.".format(data_arr_name, \
                                                                               plot_type, agg_type,
                                                                               many_to_one_agg_types))
                    # Some plot types cannot accept many-to-one aggregation.
                    if plot_type not in plot_types_handling_aggregation:
                        if agg_type not in many_to_many_agg_types:
                            raise ValueError("For the '{}' DataArray: "
                                             "the plot type '{}' only accepts many-to-many aggregation "
                                             "(currently using '{}'). Please pass any of {} as "
                                             "the aggregation type or change the plot type."
                                             .format(data_arr_name, plot_type, agg_type,
                                                     many_to_many_agg_types))

                    # Aggregate if necessary.
                    y = data_arr_plotting_data
                    if agg_type == 'min':
                        y = y.min([x_coord, y_coord])
                    if agg_type == 'mean':
                        y = y.mean([x_coord, y_coord])
                    if agg_type == 'median':
                        y = y.median([x_coord, y_coord])
                    if agg_type == 'max':
                        y = y.max([x_coord, y_coord])

                    # Handle curve fits.
                    if plot_type in plot_types_curve_fit:
                        smooth = plot_kwargs.get('smooth', True)
                        # Create the curve fit.
                        x_smooth = None if smooth else epochs_not_extrap
                        data_arr_epochs, y = get_curvefit(epochs_not_extrap, y.values, fit_type=plot_type,
                                                          x_smooth=x_smooth, fit_kwargs=plot_kwargs)
                        # Convert time stamps to NumPy datetime objects.
                        data_arr_times = np.array(list(map(_scalar_to_n64_datetime, data_arr_epochs))) \
                            if time_agg_str == 'time' else data_arr_epochs
                        # Convert the NumPy array into an xarray DataArray.
                        coords = {time_agg_str: data_arr_times}
                        dims = list(coords.keys())
                        y = xr.DataArray(y, coords=coords, dims=dims)
                    plotting_data_not_nan_and_extrap[(data_arr_name, agg_type, plot_type)] = y

    # Handle the potential for multiple plots.
    max_times_per_plot = len(times_not_all_nan_and_extrap) if max_times_per_plot is None else \
        max_times_per_plot
    num_times = len(times_not_all_nan_and_extrap)
    num_plots = int(np.ceil(num_times / max_times_per_plot))
    num_times_per_plot = round(num_times / num_plots) if num_plots != 0 else 0
    num_cols = min(num_plots, max_cols)
    num_rows = int(np.ceil(num_plots / num_cols)) if num_cols != 0 else 0
    # Set a reasonable figsize if one is not set in `fig_params`.
    fig_params.setdefault('figsize', (12 * num_cols, 6 * num_rows))
    fig = plt.figure(**fig_params) if fig is None else fig

    # Check if there are no plots to make.
    if num_plots == 0:
        return fig, plotting_data_not_nan_and_extrap

    # Create each plot. #
    for time_ind, ax_ind in zip(range(0, len(times_not_all_nan_and_extrap), num_times_per_plot),
                                range(num_plots)):
        # The time bounds of this canvas (or "Axes object" or "plot grid cell").
        ax_lower_time_bound_ind, ax_upper_time_bound_ind = \
            time_ind, min(time_ind + num_times_per_plot, len(times_not_all_nan_and_extrap))
        # Retrieve or create the axes if necessary.
        if len(times_not_all_nan_and_extrap) <= num_times_per_plot:
            fig, ax = retrieve_or_create_fig_ax(fig, ax, **fig_params)
        else:
            ax = fig.add_subplot(num_rows, num_cols, ax_ind + 1)
        ax_times_not_all_nan_and_extrap = \
            times_not_all_nan_and_extrap[ax_lower_time_bound_ind:ax_upper_time_bound_ind]
        ax_time_bounds = ax_times_not_all_nan_and_extrap[[0, -1]]
        ax_epochs = epochs[ax_lower_time_bound_ind:ax_upper_time_bound_ind]
        ax_x_locs = np_scale(ax_epochs if time_agg_str == 'time' else ax_times_not_all_nan_and_extrap)

        # Data variable plots within each plot.
        data_arr_plots = []
        legend_labels = []
        # For each data array to plot...
        for data_arr_name, agg_dict in plot_descs.items():
            # For each aggregation type (e.g. 'mean', 'median')...
            for agg_type, plot_dicts in agg_dict.items():
                # For each plot for this aggregation type...
                for plot_dict in plot_dicts:
                    for plot_type, plot_kwargs in plot_dict.items():
                        # Determine the legend label for this plot.
                        plot_type_str = \
                            {'scatter': 'scatterplot', 'line': 'lineplot',
                             'box': 'boxplot', 'gaussian': 'gaussian fit',
                             'gaussian_filter': 'gaussian filter fit',
                             'poly': 'degree {} polynomial fit',
                             'cubic_spline': 'cubic spline fit',
                             'fourier': 'Fourier fit ({} harmonics)'}[plot_type]
                        if plot_type == 'poly':
                            assert 'degree' in plot_kwargs, \
                                "For the '{}' DataArray: When using 'poly' as " \
                                "the fit type, the fit kwargs must have 'degree' " \
                                "specified.".format(data_arr_name)
                            plot_type_str = plot_type_str.format(
                                plot_kwargs.get('degree'))
                        if plot_type == 'fourier':
                            plot_type_str = plot_type_str.format(
                                plot_kwargs.get('n_harm', default_fourier_n_harm))
                        # Legend labels for the non-extrapolation
                        # and extrapolation segments
                        plot_type_strs = []

                        # Remove plot kwargs that are not recognized
                        # by plotting methods (cause errors).
                        plot_kwargs = plot_kwargs.copy()
                        plot_kwargs.pop('extrap_time', None)
                        plot_kwargs.pop('n_predict', None)
                        plot_kwargs.pop('smooth', None)
                        plot_kwargs.pop('degree', None)  # 'degree'
                        plot_kwargs.pop('n_harm', None)  # 'fourier'

                        # Handle default plot kwargs.
                        if plot_type == 'box':
                            plot_kwargs.setdefault('boxprops',
                                                   dict(facecolor='orange'))
                            plot_kwargs.setdefault('flierprops',
                                                   dict(marker='o', markersize=0.5))
                            plot_kwargs.setdefault('showfliers', False)

                        # Retrieve the plotting data.
                        y = plotting_data_not_nan_and_extrap[
                            (data_arr_name, agg_type, plot_type)]
                        y = y.sel({time_agg_str:
                                       slice(ax_time_bounds[0], ax_time_bounds[1])})

                        # Handle cases of insufficient data for this section of the plot.
                        not_nat_times = None
                        if time_agg_str == 'time':
                            not_nat_times = ~np.isnat(y[time_agg_str].values)
                        else:
                            not_nat_times = ~np.isnan(y[time_agg_str].values)
                        num_unique_times_y = len(np.unique(y[time_agg_str].values[not_nat_times]))
                        if num_unique_times_y == 0:  # There is no data.
                            continue
                        if num_unique_times_y == 1:  # There is 1 data point.
                            plot_type = 'scatter'
                            plot_kwargs = {}

                        data_arr_epochs = \
                            np.array(list(map(n64_to_epoch, y[time_agg_str].values))) \
                                if time_agg_str == 'time' else \
                                ax_times_not_all_nan_and_extrap
                        data_arr_x_locs = np.interp(data_arr_epochs,
                                                    ax_epochs, ax_x_locs)
                        data_arr_time_bounds = y[time_agg_str].values[[0, -1]]

                        # Determine if this plotting data includes extrapolated values.
                        data_arr_non_extrap_time_bounds = None
                        data_arr_has_non_extrap = \
                            data_arr_time_bounds[0] < times_not_all_nan[-1]
                        if data_arr_has_non_extrap:
                            data_arr_non_extrap_time_bounds = \
                                [data_arr_time_bounds[0], min(data_arr_time_bounds[1],
                                                              times_not_all_nan[-1])]
                            # Because the data could be smoothed, the last
                            # non-extrapolation time is the last time before
                            # or at the last non-extrapolation time
                            # for the original data.
                            non_extrap_plot_last_time = data_arr_non_extrap_time_bounds[1]
                            if num_unique_times_y > 1:
                                non_extrap_plot_last_time = \
                                    y.sel({time_agg_str: data_arr_non_extrap_time_bounds[1]},
                                          method='ffill')[time_agg_str].values
                            data_arr_non_extrap_plotting_time_bounds = [data_arr_non_extrap_time_bounds[0],
                                                                        non_extrap_plot_last_time]

                        data_arr_extrap_time_bounds = None
                        data_arr_has_extrap = times_not_all_nan[-1] < data_arr_time_bounds[1]
                        if data_arr_has_extrap:
                            data_arr_extrap_time_bounds = [max(data_arr_time_bounds[0],
                                                               extrap_times[0]),
                                                           data_arr_time_bounds[1]]
                            # Because the data could be smoothed, the first extrapolation time
                            # is the first time after the last non-extrapolation time for the original data.
                            extrap_plot_first_time = \
                                y.sel({time_agg_str: data_arr_non_extrap_time_bounds[1]},
                                      method='ffill')[time_agg_str].values \
                                    if data_arr_has_non_extrap else \
                                    data_arr_time_bounds[0]
                            data_arr_extrap_plotting_time_bounds = [extrap_plot_first_time,
                                                                    data_arr_extrap_time_bounds[1]]

                        # Separate non-extrapolation and extrapolation data.
                        if data_arr_has_non_extrap:
                            data_arr_non_extrap = \
                                y.sel({time_agg_str: slice(*data_arr_non_extrap_plotting_time_bounds)})
                            data_arr_non_extrap_epochs = \
                                np.array(list(map(n64_to_epoch, data_arr_non_extrap[time_agg_str].values))) \
                                    if time_agg_str == 'time' else data_arr_non_extrap[time_agg_str].values
                            data_arr_non_extrap_x_locs = \
                                np.interp(data_arr_non_extrap_epochs, ax_epochs, ax_x_locs)
                            # Format plotting kwargs for the non-extrapolation data.
                            plot_kwargs_non_extrap = plot_kwargs.copy()
                            plot_kwargs_non_extrap.pop('extrap_color', None)
                        if data_arr_has_extrap:
                            # Include the last non-extrapolation point so the
                            # non-extrapolation and extrapolation lines connect.
                            data_arr_extrap = \
                                y.sel({time_agg_str: slice(*data_arr_extrap_plotting_time_bounds)})
                            data_arr_extrap_epochs = \
                                np.array(list(map(n64_to_epoch, data_arr_extrap[time_agg_str].values))) \
                                    if time_agg_str == 'time' else data_arr_extrap[time_agg_str].values
                            data_arr_extrap_x_locs = \
                                np.interp(data_arr_extrap_epochs, ax_epochs, ax_x_locs)
                            # Format plotting kwargs for the extrapolation data.
                            plot_kwargs_extrap = plot_kwargs.copy()
                            extrap_color = plot_kwargs_extrap.pop('extrap_color', None)
                            if extrap_color is not None:
                                plot_kwargs_extrap['color'] = extrap_color

                        # Specify non-extrap and extrap plotting args.
                        if data_arr_has_non_extrap:
                            plot_args_non_extrap = \
                                [data_arr_non_extrap_x_locs, data_arr_non_extrap]
                        if data_arr_has_extrap:
                            plot_args_extrap = \
                                [data_arr_extrap_x_locs, data_arr_extrap]

                        # Actually create the plot.
                        def create_plot(x_locs, data_arr, **plot_kwargs):
                            """
                            Creates a plot

                            Parameters
                            ----------
                            x_locs: xarray.DataArray
                                A 1D `xarray.DataArray` containing ascending values
                                in range [0,1], denoting the x locations on the current
                                canvas at which to plot data with corresponding time
                                indicies in `data_arr`.
                            data_arr: xarray.DataArray
                                An `xarray.DataArray` containing a dimension named
                                `time_agg_str` (the value of that variable in this context).

                            Returns
                            -------
                            plot_obj: matplotlib.artist.Artist
                                The plot.
                            """
                            plot_obj = None
                            if plot_type == 'scatter':
                                data_arr_dims = list(data_arr.dims)
                                data_arr_flat = data_arr.stack(flat=data_arr_dims)
                                plot_obj = ax.scatter(x_locs, data_arr_flat)
                            elif plot_type in ['line', 'gaussian', 'gaussian_filter',
                                               'poly', 'cubic_spline', 'fourier']:
                                plot_obj = ax.plot(x_locs, data_arr)[0]
                            elif plot_type == 'box':
                                boxplot_nan_mask = ~np.isnan(data_arr)
                                # Data formatted for matplotlib.pyplot.boxplot().
                                filtered_formatted_data = []
                                for i, (d, m) in enumerate(zip(data_arr.values,
                                                               boxplot_nan_mask.values)):
                                    if len(d[m] != 0):
                                        filtered_formatted_data.append(d[m])
                                box_width = 0.5 * np.min(np.diff(x_locs)) \
                                    if len(x_locs) > 1 else 0.5
                                # `manage_xticks=False` to avoid excessive padding on x-axis.
                                bp = ax.boxplot(filtered_formatted_data,
                                                widths=[box_width] * len(filtered_formatted_data),
                                                positions=x_locs, patch_artist=True,
                                                manage_xticks=False, **plot_kwargs)
                                plot_obj = bp['boxes'][0]
                            return plot_obj

                        if data_arr_has_non_extrap:
                            plot_obj = create_plot(*plot_args_non_extrap, **plot_kwargs_non_extrap)
                            data_arr_plots.append(plot_obj)
                            plot_type_strs.append(plot_type_str)
                        if data_arr_has_extrap and plot_type in plot_types_supporting_extrapolation:
                            plot_obj = create_plot(*plot_args_extrap, **plot_kwargs_extrap)
                            data_arr_plots.append(plot_obj)
                            plot_type_strs.append('extrapolation of ' + plot_type_str)
                        plot_type_str_suffix = ' of {}'.format(agg_type) if agg_type != 'none' else ''
                        plot_type_strs = [plot_type_str + plot_type_str_suffix
                                          for plot_type_str in plot_type_strs]
                        [legend_labels.append('{} of {}'.format(plot_type_str, data_arr_name))
                         for plot_type_str in plot_type_strs]

        # Label the axes and create the legend.
        date_strs = \
            np.array(list(map(lambda time: np_dt64_to_str(time), ax_times_not_all_nan_and_extrap))) \
                if time_agg_str == 'time' else \
                naive_months_ticks_by_week(ax_times_not_all_nan_and_extrap) \
                    if time_agg_str in ['week', 'weekofyear'] else \
                    month_ints_to_month_names(ax_times_not_all_nan_and_extrap)
        plt.xticks(ax_x_locs, date_strs, rotation=45, ha='right', rotation_mode='anchor')
        if show_legend:
            ax.legend(handles=data_arr_plots, labels=legend_labels, loc='best')
        title_postpend = " ({} to {})".format(date_strs[0], date_strs[-1])
        title_prepend = "Figure {}".format(ax_ind) if title is None else title
        ax.set_title(title_prepend + title_postpend)
    return fig, plotting_data_not_nan_and_extrap


## Curve fitting ##

def get_curvefit(x, y, fit_type, x_smooth=None, n_pts=n_pts_smooth, fit_kwargs=None):
    """
    Gets a curve fit given x values, y values, a type of curve, and parameters for that curve.

    Parameters
    ----------
    x: np.ndarray
        A 1D NumPy array. The x values to fit to.
    y: np.ndarray
        A 1D NumPy array. The y values to fit to.
    fit_type: str
        The type of curve to fit. One of ['gaussian', 'gaussian_filter', 'poly',
                                          'cubic_spline', 'fourier'].
        The option 'gaussian' creates a Gaussian fit.
        The option 'gaussian_filter' creates a Gaussian filter fit.
        The option 'poly' creates a polynomial fit.
        The option 'cubic_spline' creates a cubic spline fit.
        The option 'fourier' creates a Fourier curve fit.
    x_smooth: list-like
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    fit_kwargs: dict
        Keyword arguments for the selected fit type.
        In the case of `fit_type == 'poly'`, this must contain a 'degree' entry (an int),
        which is the degree of the polynomial to fit.
        In the case of `fit_type == 'gaussian_filter'`, this may contain a 'sigma' entry,
        which is the standard deviation of the Gaussian kernel.
        A larger value yields a smoother but less close-fitting curve.
        In the case of `fit_type == 'fourier'`, this may contain 'n_predict' or 'n_harm' entries.
        The 'n_predict' entry is the number of points to extrapolate.
        The points will be spaced evenly by the mean spacing of values in `x`.
        The 'n_harm' entry is the number of harmonics to use.
        A higher value yields a closer fit.

    Returns
    -------
    x_smooth, y_smooth: numpy.ndarray
        The smoothed x and y values of the curve fit.
        If there are no non-NaN values in `y`, these will be filled with `n_pts` NaNs.
        If there is only 1 non-NaN value in `y`, these will be filled with
        their corresponding values (y or x value) for that point to a length of `n_pts`.

    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    interpolation_curve_fits = ['gaussian', 'gaussian_filter',
                                'poly', 'cubic_spline']
    extrapolation_curve_filts = ['fourier']
    # Handle NaNs (omit them).
    not_nan_mask = ~np.isnan(y)
    x = x[not_nan_mask]; y = y[not_nan_mask]

    # Handle the cases of there being too few points to curve fit.
    if len(y) == 0:
        x_smooth = np.repeat(np.nan, n_pts)
        y_smooth = np.repeat(np.nan, n_pts)
        return x_smooth, y_smooth
    if len(y) == 1:
        x_smooth = np.repeat(x[0], n_pts)
        y_smooth = np.repeat(y[0], n_pts)
        return x_smooth, y_smooth

    if x_smooth is None:
        x_smooth_inds = np.linspace(0, len(x) - 1, n_pts)
        x_smooth = np.interp(x_smooth_inds, np.arange(len(x)), x)

    opt_params = {}
    if fit_type == 'gaussian':
        x_smooth, y_smooth = gaussian_fit(x, y, x_smooth)
    elif fit_type == 'gaussian_filter':
        if 'sigma' in fit_kwargs:
            opt_params.update(dict(sigma=fit_kwargs.get('sigma')))
        x_smooth, y_smooth = gaussian_filter_fit(x, y, x_smooth,
                                                 **opt_params)
    elif fit_type == 'poly':
        assert 'degree' in fit_kwargs.keys(), \
            "When plotting a polynomal fit, there must be" \
            "a 'degree' entry in the plot_kwargs parameter."
        degree = fit_kwargs.get('degree')
        x_smooth, y_smooth = poly_fit(x, y, degree, x_smooth)
    elif fit_type == 'cubic_spline':
        cs = CubicSpline(x, y)
        y_smooth = cs(x_smooth)
    if fit_type in extrapolation_curve_filts:
        n_predict = fit_kwargs.get('n_predict', 0)
        if fit_type == 'fourier':
            if 'n_harm' in fit_kwargs:
                opt_params.update(dict(n_harm=fit_kwargs.get('n_harm')))
            x_smooth, y_smooth = \
                fourier_fit(x, y, n_predict, x_smooth,
                            **opt_params)
    return x_smooth, y_smooth


## End curve fitting ##


#     ## Color utils ##


def convert_name_rgb_255(color):
    """
    Converts a name of a matplotlib color to a list of rgb values in the range [0,255].
    Else, returns the original argument.

    Parameters
    ----------
    color: str
        The color name to convert to an rgb list.
    """
    return [int(255 * rgb) for rgb in mpl.colors.to_rgb(color)] if isinstance(color, str) else color


def convert_name_rgba_255(color):
    """
    Converts a name of a matplotlib color to a list of rgba values in the range [0,255].
    Else, returns the original argument.

    Parameters
    ----------
    color: str
        The color name to convert to an rgba list.
    """
    return [*convert_name_rgb_255(color), 255] if isinstance(color, str) else color


def norm_color(color):
    """
    Converts either a string name of a matplotlib color or a 3-tuple of rgb values
    in the range [0,255] to a 3-tuple of rgb values in the range [0,1].

    Parameters
    ----------
    color: str or list-like of numeric
        The name of a matplolib color or a .
    """
    color = convert_name_rgb_255(color)
    if len(color) == 3:
        color = [rgb / 255 for rgb in color]
    return color


## End color utils ##

## Matplotlib colormap functions ##

def create_discrete_color_map(data_range=None, colors=None, cmap=None,
                              th=None, pts=None, cmap_name='my_cmap',
                              data_range_fmt=None, pts_fmt=None):
    """
    Creates a discrete matplotlib LinearSegmentedColormap with thresholds for color changes.
    Exclusively either `colors` or `cmap` must be specified (i.e. one and only one).
    At least one of the parameters `th` or `pts` may be specified, but not both.

    Parameters
    ----------
    data_range: list
        A 2-tuple of the minimum and maximum values the data may take.
        Can be omitted if `pts` is specified as a list-like of points.
    colors: list-like
        Colors to use between thresholds specified in `th` or around points specified in `pts`.
        Colors can be string names of matplotlib colors, 3-tuples of rgb values in range [0,255],
        or 4-tuples of rgba values in range [0,1].
    cmap: matplotlib.colors.Colormap
        A matplotlib colormap used to color data in the regions between thresholds
        specified in `th` or around points specified in `pts`.
    th: list-like of float
        Threshold values separating colors, so `len(colors) == len(th)+1`.
        Must be in the range of `data_range` - noninclusive.
    pts: int or list-like of float
        Points around which to color the same. This can be either an integer
        specifying the number of evenly-spaced points to use or a list-like of points,
        in which case values must be in the range of `data_range` - inclusive.
        The thresholds used will be the midpoints between points in `pts`.
    cmap_name: str
        The name of the created colormap for matplotlib.
    data_range_fmt: list-like of size 2
        A mutable container intended to hold values used to set vmin and vmax, respectively, of
        `pyplot.imshow()` for the purpose of formatting a colorbar. Only useful if `pts` is
        specified as a list-like.
    pts_fmt: list-like
        A mutable container intended to hold the midpoints of the thresholds. This must have the same length
        as the number of points specified by `pts` or have a length of `len(th)+1`.

    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    assert (colors is None) ^ (cmap is None), \
        "Exclusively either `colors` or `cmap` must be specified."
    assert th is None or pts is None, \
        "The parameters `th` or `pts` may be specified, but not both."

    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    if th is None:  # If `th` is not supplied, construct it based on other arguments.
        if pts is not None:
            if isinstance(pts, int):  # Use `pts` as the number of evenly-spaced points.
                assert pts > 0, "The number of points specified by `pts` must be positive."
                th_spacing = (data_range[1] - data_range[0]) / pts
                th = np.linspace(data_range[0] + th_spacing, data_range[1] - th_spacing, pts - 1)
            else:  # Use `pts` as a list-like of points to put thresholds between.
                assert data_range[0] <= min(pts) and max(pts) <= data_range[1], \
                    "The values in `pts` must be within `data_range`, inclusive."
                assert len(pts) > 0, "The parameter `pts` is a list, but it has no elements. " \
                                     "Please ensure it has one or more numeric elements."
                if len(pts) == 1:
                    th = []
                elif len(pts) > 1:
                    # Choose imaginary lower and upper bounds of the data to scale `pts` with
                    # so that the first and last color regions are sized appropriately.
                    data_range_fmt = [None] * 2 if data_range_fmt is None else data_range_fmt
                    data_range_fmt[0] = pts[0] - (pts[1] - pts[0]) / 2
                    data_range_fmt[1] = pts[-1] + (pts[-1] - pts[-2]) / 2
                    pts = np.interp(pts, data_range_fmt, data_range)  # (0,1))
                    th = [pts[ind - 1] + (pts[ind] - pts[ind - 1]) / 2 for ind in range(1, len(pts))]
        else:
            assert colors is not None, \
                "If neither `th` nor `pts` are specified, `colors` must be specified."
            th_spacing = (data_range[1] - data_range[0]) / len(colors)
            th = np.linspace(data_range[0] + th_spacing, data_range[1] - th_spacing, len(colors) - 1)
    else:
        assert len(th) == 0 or (data_range[0] < min(th) and max(th) < data_range[1]), \
            "The values in `th` must be within `data_range`, exclusive."
        # Normalize threshold values based on the data range.
        th = [(val - data_range[0]) / (data_range[1] - data_range[0]) for val in th]
    th = np.interp(th, data_range, (0, 1))
    th = [0.0] + list(th) + [1.0]
    if pts_fmt is not None:
        for ind in range(len(th) - 1):
            pts_fmt[ind] = th[ind] + (th[ind + 1] - th[ind]) / 2

    if colors is None:  # If `colors` is not supplied, construct it based on other arguments.
        assert cmap is not None, \
            "If `colors` is not specified, `cmap` must be specified."
        colors = [cmap(th[ind - 1] + (th[ind] - th[ind - 1]) / 2) for ind in range(1, len(th))]
    else:
        colors = list(map(norm_color, colors))

    cdict = {}
    # These are fully-saturated red, green, and blue - not the matplotlib colors for 'red', 'green', and 'blue'.
    primary_colors = ['red', 'green', 'blue']
    # Get the 3-tuples of rgb values for the colors.
    color_rgbs = [(mpl.colors.to_rgb(color) if isinstance(color, str) else color) for color in colors]
    # For each color entry to go into the color dictionary...
    for primary_color_ind, primary_color in enumerate(primary_colors):
        cdict_entry = [None] * len(th)
        # For each threshold (as well as 0.0 and 1.0), specify the values for this primary color.
        for row_ind, th_ind in enumerate(range(len(th))):
            # Get the two colors that this threshold corresponds to.
            th_color_inds = [0, 0] if th_ind == 0 else \
                [len(colors) - 1, len(colors) - 1] if th_ind == len(th) - 1 else \
                    [th_ind - 1, th_ind]
            primary_color_vals = [color_rgbs[th_color_ind][primary_color_ind] for th_color_ind in th_color_inds]
            cdict_entry[row_ind] = (th[th_ind],) + tuple(primary_color_vals)
        cdict[primary_color] = cdict_entry
    cmap = LinearSegmentedColormap(cmap_name, cdict)
    return cmap


def create_gradient_color_map(data_range, colors, positions=None, cmap_name='my_cmap'):
    """
    Creates a gradient colormap with a LinearSegmentedColormap. Currently only creates linear gradients.

    Parameters
    ----------
    data_range: list-like
        A 2-tuple of the minimum and maximum values the data may take.
    colors: list of str or list of tuple
        Colors can be string names of matplotlib colors or 3-tuples of rgb values in range [0,255].
        The first and last colors are placed at the beginning and end of the colormap, respectively.
    positions: list-like
        The values which are colored with corresponding colors in `colors`,
        except the first and last colors, so `len(positions) == len(colors)-2`.
        Positions must be in the range of `data_range` - noninclusive.
        If no positions are provided, the colors are evenly spaced.
    cmap_name: str
        The name of the created colormap for matplotlib.

    Examples
    --------
    Creating a linear gradient colormap of red, green, and blue, with even spacing between them:
        create_gradient_color_map(data_range=(0,1), positions=(0.5,), colors=('red', 'green', 'blue'))
    Which can also be done without specifying `positions`:
        create_gradient_color_map(data_range=(0,1), colors=('red', 'green', 'blue'))
    """
    # Normalize position values based on the data range.
    if positions is None:
        range_size = data_range[1] - data_range[0]
        spacing = range_size / (len(colors) - 1)
        positions = [spacing * i for i in range(1, len(colors) - 1)]
    else:
        positions = list(map(lambda val: (val - data_range[0]) / (data_range[1] - data_range[0]), positions))

    colors = list(map(norm_color, colors))  # Normalize color values for colormap creation.
    positions = [0.0] + positions + [1.0]

    cdict = {}
    # These are fully-saturated red, green, and blue - not the matplotlib colors for 'red', 'green', and 'blue'.
    primary_colors = ['red', 'green', 'blue']
    # Get the 3-tuples of rgb values for the colors.
    color_rgbs = [(mpl.colors.to_rgb(color) if isinstance(color, str) else color) for color in colors]
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(positions, color_rgbs):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    return LinearSegmentedColormap(cmap_name, cdict)

## End matplotlib colormap functions ##

## Misc ##

def get_ax_size(fig, ax):
    """
    Given matplotlib Figure (fig) and Axes (ax) objects, return
    the width and height of the Axes object in inches as a list.
    """
    # Credit goes to https://stackoverflow.com/a/19306776/5449970.
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return [bbox.width, bbox.height]


def xarray_imshow(data, x_coord='longitude', y_coord='latitude', width=10,
                  fig=None, ax=None, use_colorbar=True, cbar_labels=None,
                  use_legend=False, legend_labels=None, fig_kwargs=None,
                  imshow_kwargs=None, x_label_kwargs=None, y_label_kwargs=None,
                  cbar_kwargs=None, nan_color='white', legend_kwargs=None,
                  ax_tick_label_kwargs=None, x_tick_label_kwargs=None,
                  y_tick_label_kwargs=None, title=None, title_kwargs=None,
                  possible_plot_values=None):
    """
    Shows a heatmap of an xarray DataArray with only latitude and longitude dimensions.
    Unlike matplotlib `imshow()` or `data.plot.imshow()`, this sets axes ticks and labels.
    It also simplifies creating a colorbar and legend.

    Parameters
    ----------
    data: xarray.DataArray
        The xarray.DataArray containing only latitude and longitude coordinates.
    x_coord, y_coord: str
        Names of the x and y coordinates in `data` to use as tick and axis labels.
    width: numeric
        The width of the created ``matplotlib.figure.Figure``, if none is supplied in `fig`.
        The height will be set to maintain aspect ratio.
        Will be overridden by `'figsize'` in `fig_kwargs`, if present.
    fig: matplotlib.figure.Figure
        The figure to use for the plot.
        If `ax` is not supplied, the Axes object used will be the first.
    ax: matplotlib.axes.Axes
        The axes to use for the plot.
    use_colorbar: bool
        Whether or not to create a colorbar to the right of the axes.
    cbar_labels: list
        A list of strings to label the colorbar.
    use_legend: bool
        Whether or not to create a legend showing labels for unique values.
        Only use if you are sure you have a low number of unique values.
    legend_labels: dict
        A mapping of values to legend labels.
    fig_kwargs: dict
        The dictionary of keyword arguments used to build the figure.
    imshow_kwargs: dict
        The dictionary of keyword arguments passed to `plt.imshow()`.
        You can pass a colormap here with the key 'cmap'.
    x_label_kwargs, y_label_kwargs: dict
        Dictionaries of keyword arguments for
        `Axes.set_xlabel()` and `Axes.set_ylabel()`, respectively.
        They cannot reference the same dictionary.
    cbar_kwargs: dict
        The dictionary of keyword arguments passed to `plt.colorbar()`.
        Some parameters of note include 'ticks', which is a list of values to place ticks at.
    nan_color: str or list-like
        The color used for NaN regions. Can be a string name of a matplotlib color or
        a 3-tuple (list-like) of rgb values in range [0,255].
    legend_kwargs: dict
        The dictionary of keyword arguments passed to `plt.legend()`.
    ax_tick_label_kwargs: dict
        The dictionary of keyword arguments passed to `ax.tick_params()`.
    x_tick_label_kwargs, y_tick_label_kwargs: dict
        Dictionaries of keyword arguments passed to `ax.set_xticklabels()`
        and `ax.set_yticklabels()`, respectively.
    title: str
        The title of the figure.
    title_kwargs: dict
        The dictionary of keyword arguments passed to `ax.set_title()`.
    possible_plot_values: list-like
        The possible range of values for `data`. The affects the coloring of the map and the legend entries.

    Returns
    -------
    fig, ax, im, cbar: matplotlib.figure.Figure, matplotlib.axes.Axes,
                       matplotlib.image.AxesImage,  matplotlib.colorbar.Colorbar
        The figure and axes used as well as the image returned by `pyplot.imshow()` and the colorbar.
        If `use_colorbar == False`, `cbar` will be `None`.

    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Figure kwargs
    # Use `copy()` to avoid modifying the original dictionaries.
    fig_kwargs = {} if fig_kwargs is None else fig_kwargs.copy()
    figsize = \
        fig_kwargs.setdefault('figsize', figure_ratio(data, x_coord, y_coord,
                                                      fixed_width=width))
    # Imshow kwargs
    imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs.copy()
    imshow_kwargs.setdefault('interpolation', 'nearest')

    nan_color = norm_color(nan_color)  # Normalize color value for matplotlib.

    fig, ax = retrieve_or_create_fig_ax(fig, ax, **fig_kwargs)
    axsize = get_ax_size(fig, ax)  # Scale fonts on axis size, not figure size.

    # Axis label kwargs
    x_label_kwargs = {} if x_label_kwargs is None else x_label_kwargs.copy()
    y_label_kwargs = {} if y_label_kwargs is None else y_label_kwargs.copy()

    # Axis tick label kwargs
    ax_tick_label_kwargs = {} if ax_tick_label_kwargs is None else \
        ax_tick_label_kwargs.copy()
    x_tick_label_kwargs = {} if x_tick_label_kwargs is None else \
        x_tick_label_kwargs
    y_tick_label_kwargs = {} if y_tick_label_kwargs is None else \
        y_tick_label_kwargs

    # Handle display of NaN values.
    data_arr = data.values
    masked_array = np.ma.array(data_arr, mask=np.isnan(data_arr))
    cmap = imshow_kwargs.setdefault('cmap', plt.get_cmap('viridis'))
    cmap.set_bad(nan_color)
    # Handle kwargs for `imshow()`.
    vmin, vmax = (np.min(possible_plot_values), np.max(possible_plot_values)) \
                  if possible_plot_values is not None else (np.nanmin(data), np.nanmax(data))
    imshow_kwargs.setdefault('vmin', vmin)
    imshow_kwargs.setdefault('vmax', vmax)
    im = ax.imshow(masked_array, **imshow_kwargs)

    # Set axis labels and tick labels.
    xarray_set_axes_labels(data, ax, x_coord, y_coord,
                           x_label_kwargs, y_label_kwargs,
                           ax_tick_label_kwargs,
                           x_tick_label_kwargs, y_tick_label_kwargs)

    # Set the title.
    if title is not None:
        title_kwargs = {} if title_kwargs is None else title_kwargs.copy()
        ax.set_title(title, **title_kwargs)

    # Create a colorbar.
    if use_colorbar:
        cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs.copy()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7.5%", pad=0.05)
        cbar = fig.colorbar(im, ax=ax, cax=cax, **cbar_kwargs)
        if cbar_labels is not None:
            cbar.ax.set_yticklabels(cbar_labels)
    else:
        cbar = None

    # Create a legend.
    if use_legend:
        legend_kwargs = {} if legend_kwargs is None else legend_kwargs.copy()
        legend_kwargs.setdefault("framealpha", 0.4)

        # Determine the legend labels. If no set of values to create legend entries for
        # is specified, use the unique values.
        if possible_plot_values is None:
            legend_values = np.unique(data.values)
            legend_values = legend_values[~np.isnan(legend_values)]
        else:
            legend_values = possible_plot_values
        if legend_labels is None:
            legend_labels = ["{}".format(value) for value in legend_values]
        else:
            legend_labels = [legend_labels.get(value, "{}".format(value)) for value in legend_values]

        colors = [im.cmap(value/np.max(legend_values)) for value in legend_values]
        patches = [mpatches.Patch(color=colors[i], label=legend_labels[i])
                   for i in range(len(legend_values))]

        legend_kwargs.setdefault('loc', 'best')
        legend_kwargs['handles'] = patches
        ax.legend(**legend_kwargs)

    return fig, ax, im, cbar


def xarray_set_axes_labels(data, ax, x_coord='longitude', y_coord='latitude',
                           x_label_kwargs=None, y_label_kwargs=None,
                           ax_tick_label_kwargs=None,
                           x_tick_label_kwargs=None, y_tick_label_kwargs=None):
    """
    Sets tick locations and labels for x and y axes on a `matplotlib.axes.Axes`
    object such that the tick labels do not overlap. This currently only supports
    numeric coordinates.

    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
        The xarray Dataset or DataArray containing latitude and longitude coordinates.
    ax: matplotlib.axes.Axes
        The matplotlib Axes object to set tick locations and labels for.
    x_coord, y_coord: str
        Names of the x and y coordinates in `data` to use as tick and axis labels.
    x_label_kwargs, y_label_kwargs: dict
        Dictionaries of keyword arguments for
        `Axes.set_xlabel()` and `Axes.set_ylabel()`, respectively.
    ax_tick_label_kwargs: dict
        The dictionary of keyword arguments passed to `ax.tick_params()`.
    x_tick_label_kwargs, y_tick_label_kwargs: dict
        Dictionaries of keyword arguments passed to `ax.set_xticklabels()`
        and `ax.set_yticklabels()`, respectively.

    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    import string
    # Avoid modifying the original arguments.
    x_label_kwargs = {} if x_label_kwargs is None else x_label_kwargs.copy()
    y_label_kwargs = {} if y_label_kwargs is None else y_label_kwargs.copy()
    ax_tick_label_kwargs = {} if ax_tick_label_kwargs is None else \
        ax_tick_label_kwargs.copy()
    x_tick_label_kwargs = {} if x_tick_label_kwargs is None else \
        x_tick_label_kwargs.copy()
    y_tick_label_kwargs = {} if y_tick_label_kwargs is None else \
        y_tick_label_kwargs.copy()

    width, height = get_ax_size(ax.figure, ax)

    # Labels
    x_label_kwargs.setdefault('xlabel', x_coord)
    ax.set_xlabel(**x_label_kwargs)
    y_label_kwargs.setdefault('ylabel', y_coord)
    ax.set_ylabel(**y_label_kwargs)

    # Tick labels
    ax.tick_params(**ax_tick_label_kwargs)
    # X ticks
    x_vals = data[x_coord].values
    x_fontsize = \
        x_tick_label_kwargs.setdefault('fontsize', mpl.rcParams['font.size'])
    label_every = max(1, int(round(1 / 10 * len(x_vals) * x_fontsize / width)))
    x_labels = ["{0:.4f}".format(float(x_val)) for x_val in x_vals[::label_every]]
    ax.set_xticks(range(len(x_vals))[::label_every])
    x_tick_label_kwargs.setdefault('rotation', 30)
    ax.set_xticklabels(x_labels, **x_tick_label_kwargs)
    # Y ticks
    y_vals = data[y_coord].values
    y_fontsize = \
        y_tick_label_kwargs.setdefault('fontsize', mpl.rcParams['font.size'])
    label_every = max(1, int(round(1 / 10 * len(y_vals) * y_fontsize / height)))
    y_labels = ["{0:.4f}".format(float(y_val)) for y_val in y_vals[::label_every]]
    ax.set_yticks(range(len(y_vals))[::label_every])
    ax.set_yticklabels(y_labels, **y_tick_label_kwargs)


def figure_ratio(data, x_coord='longitude', y_coord='latitude',
                 fixed_width=None, fixed_height=None,
                 num_cols=1, num_rows=1):
    """
    Returns a list of the width and height that match constraints on height
    and width for a figure while maintaining aspect ratio if possible.
    Also can be used to size a figure of a grid of plots of identically sized cells.
    Specifically, the width and height are scaled by `num_cols` and `num_rows`.

    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray or list-like
        Can be either of the following:
        1. A list-like of x and y dimension sizes, respectively
        2. An xarray Dataset or DataArray containing x and y dimensions
    x_coord, y_coord: str
        Names of the x and y coordinates in `data`.
    fixed_width, fixed_height: float
        The desired width or height. If both are specified, the aspect
        ratio is maintained and `fixed_width` and `fixed_height` are
        treated as maximum values for the size of a single grid element.
    num_cols, num_rows: int
        The number of columns and rows in the grid the plots will be in.
        Zero, one, or both may be specified.
    """
    assert (fixed_width is not None) or (fixed_height is not None), \
        "At least one of `fixed_width` or `fixed_height` must be specified."
    # Determine the x and y dimension sizes and the aspect ratio.
    if isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray):
        x_sz, y_sz = len(data[x_coord]), len(data[y_coord])
    else:
        x_sz, y_sz = data[0], data[1]
    aspect_ratio = y_sz / x_sz
    # Determine the figure size.
    if fixed_width is not None:
        width = fixed_width
        height = width * aspect_ratio
    elif fixed_height is not None:
        height = fixed_height
        width = height / aspect_ratio
    # If both `fixed_width` and `fixed_height` are specified, treat as maximums.
    if (fixed_width is not None) and (fixed_height is not None):
        if width > fixed_width:
            height *= fixed_width / width
            width = fixed_width
        if height > fixed_height:
            width *= fixed_height / height
            height = fixed_height
    return [width * num_cols, height * num_rows]


def retrieve_or_create_fig_ax(fig=None, ax=None, **subplots_kwargs):
    """
    Returns appropriate matplotlib Figure and Axes objects given Figure and/or Axes objects.
    If neither is supplied, a new figure will be created with associated axes.
    If only `fig` is supplied, `(fig,fig.axes[0])` is returned. That is, the first Axes object will be used (and created if necessary).
    If `ax` is supplied, `(fig, ax)` is returned.

    Returns
    -------
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and the axes of that figure.
    **subplots_kwargs: dict
        A dictionary of keyword arguments to passed to `matplotlib.pyplot.subplots()`,
        such as `ncols` or `figsize`.
    """
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(**subplots_kwargs)
        else:
            if len(fig.axes) == 0:
                fig.add_axes([1, 1, 1, 1])
            ax = fig.axes[0]
    return fig, ax


def remove_non_unique_ordered_list_str(ordered_list):
    """
    Sets all occurrences of a value in an ordered list after its first occurence to ''.
    For example, ['a', 'a', 'b', 'b', 'c'] would become ['a', '', 'b', '', 'c'].
    """
    prev_unique_str = ""
    for i in range(len(ordered_list)):
        current_str = ordered_list[i]
        if current_str != prev_unique_str:
            prev_unique_str = current_str
        else:
            ordered_list[i] = ""
    return ordered_list


# For February, assume leap years are included.
days_per_month = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                  7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def get_weeks_per_month(num_weeks):
    """
    Including January, give 5 weeks to every third month - accounting for 
    variation between 52 and 54 weeks in a year by adding weeks to the last 3 months.
    """
    last_months_num_weeks = None
    if num_weeks <= 52:
        last_months_num_weeks = [5, 4, 4]
    elif num_weeks == 53:
        last_months_num_weeks = [5, 4, 5]
    elif num_weeks == 54:
        last_months_num_weeks = [5, 5, 5]
    return {month_int: num_weeks for (month_int, num_weeks) in
            zip(days_per_month.keys(), [5, 4, 4] * 3 + last_months_num_weeks)}


num_weeks_per_month = np.tile([5, 4], 6)
last_week_int_per_month = np.cumsum(num_weeks_per_month)
month_names_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_names_long = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']


def day_of_year_int_to_str(day):
    """
    Converts an integer day of year to a string containing the month and day, like "January 1".
    The argument value must be in range [1,366].

    Parameters
    ---------
    day: int
        The day of the year, represented as an integer.
    """
    month_int = 1
    while month_int < 12:
        days_curr_month = days_per_month[month_int]
        if day < days_curr_month:
            break
        else:
            day -= days_curr_month
            month_int += 1
    month_name = month_names_long[month_int - 1]
    return "{} {}".format(month_name, day)


def month_ints_to_month_names(month_ints):
    """
    Converts ordinal numbers for months (in range [1,12]) to their 3-letter names.
    """
    return [month_names_short[i - 1] for i in month_ints]


def week_int_to_month_name(week_int):
    month_ind = np.argmax(week_int <= last_week_int_per_month)
    return month_names_short[month_ind]


def week_ints_to_month_names(week_ints):
    return [week_int_to_month_name(week_int) for week_int in week_ints]


def naive_months_ticks_by_week(week_ints=None):
    """
    Given a list of week numbers (in range [1,54]), returns a list of month strings separated by spaces.
    Covers 54 weeks if no list-like of week numbers is given.
    This is only intended to be used for labeling axes in plotting.
    """
    month_ticks_by_week = []
    week_ints = list(range(1, 55)) if week_ints is None else week_ints
    month_ticks_by_week = remove_non_unique_ordered_list_str(week_ints_to_month_names(week_ints))
    return month_ticks_by_week
