# MIT License

# Copyright (c) 2017 Jacob Bourne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_rolling_regression(data, width, bmark_col, manager_cols, rf_col=None, **kwargs):
    """
    create a chart of rolling regressions
    :param data: the data
    :param width: the width
    :param bmark_col: the benchmark column
    :param manager_cols: one or more managers
    :param rf_col: rf_col (defaults to None)
    :param kwargs: optional arguments
    :return: the chart
    """

    if not isinstance(manager_cols, Iterable):
        manager_cols = [manager_cols]

    # the series for the benchmark
    bmark_series = data[data.columns[bmark_col]].dropna()
    bmark_name = data.columns[bmark_col]

    legend_names = []

    f, axarr = plt.subplots(3, sharex=True, figsize=kwargs.pop('figsize', (8, 6)))

    for manager_col in manager_cols:
        # the manager series
        manager_series = data[data.columns[manager_col]].dropna()
        manager_name = data.columns[manager_col]

        data_series = pd.concat([bmark_series, manager_series], axis=1, join='inner')

        # now handle the rF rate
        if rf_col is None:
            rf_data = [0.0] * len(data_series)
            rf_series = pd.Series(rf_data, data_series.index)
        else:
            rf_series = data[data.columns[rf_col]]

        # join everything
        data_series = pd.concat([data_series, rf_series], axis=1, join='inner')

        # now create the actual series
        adj_bmark = data_series[data_series.columns[0]].values - data_series[data_series.columns[2]].values
        adj_manager = data_series[data_series.columns[1]].values - data_series[data_series.columns[2]].values
        adj_dict = {bmark_name: adj_bmark, manager_name: adj_manager}

        st_series = pd.DataFrame.from_dict(adj_dict, orient='columns')
        st_series.index = data_series.index
        st_series = st_series[[bmark_name, manager_name]]

        legend_names.append(' vs '.join([manager_name, bmark_name]))

        # finally
        results = roll(st_series, width).apply(calc_reg)

        # we can finally chart the stuff
        axarr[0].plot(results['alpha'])
        axarr[1].plot(results['beta'])
        axarr[2].plot(results['r2'])

    # pretty it up
    f.suptitle(kwargs.pop('title', 'Rolling Regression Summary'))
    # axis titles
    axarr[0].set_ylabel("Alpha")
    axarr[1].set_ylabel("Beta")
    axarr[2].set_ylabel("R-Squared")
    axarr[2].set_xlabel("Date")
    # legend
    plt.figlegend(axarr[0].lines, legend_names, loc=4)

    return plt


def roll(df, w, **kwargs):
    """
    custom roll function to enable multiple columns to be selected in the window
    :param df: the data frame
    :param w: the width
    :param kwargs: any other arguments
    :return: a multi column window
    """
    roll_array = np.dstack([df.values[i:i + w, :] for i in range(len(df.index) - w + 1)]).T
    panel = pd.Panel(roll_array,
                     items=df.index[w - 1:],
                     major_axis=df.columns,
                     minor_axis=pd.Index(range(w), name='roll'))
    return panel.to_frame().unstack().T.groupby(level=0, **kwargs)


def calc_reg(series):
    """
    calculate regression statistics and return a pd.Series of alpha, beta and r2
    :param series: the series
    :return: pd.Series
    """
    x = series[series.columns[0]].values
    y = series[series.columns[1]].values
    A = np.vstack([x, np.ones(len(x))]).T

    m, c = np.linalg.lstsq(A, y)[0]
    r2 = np.corrcoef(x, y)[0][1] ** 2

    return pd.Series({'beta': m, 'alpha': c, 'r2': r2})
