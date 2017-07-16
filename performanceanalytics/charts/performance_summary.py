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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import performanceanalytics.statistics as pas
import os


def create_performance_summary(data, manager_col=0, other_cols=None, **kwargs):
    """
    create a three paneled chart, top panel is cumulative return for the manager and any
    other series you want.
    :param data: the raw data
    :param manager_col: the column with the main manager (defaults to 0)
    :param other_cols: array of columns with other series to chart (defaults to None)
    :param kwargs: other overwrides for chart
    :return: the chart
    """
    # create the grid
    f = plt.figure(figsize=kwargs.pop('figsize',(8,6)))
    # set height ratios for sublots
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax2 = plt.subplot(gs[2], sharex=ax1)

    # the first chart
    ax1_cols = [manager_col]
    if other_cols is not None:
        for oc in other_cols:
            ax1_cols.append(oc)

    df1 = data[data.columns[ax1_cols]]
    cp = ((1 + df1).cumprod()) - 1
    ax0.plot(cp)

    # the second chart
    df2 = data[data.columns[manager_col]]
    ax1.bar(df2.index.values, df2.values, align="center", width=20)

    # the third chart
    dd_series = df1.apply(dd, 0)
    ax2.plot(dd_series)

    # now pretty it up

    # title and legend
    f.suptitle(kwargs.pop('title', '{} Rolling Performance Summary'.format(data.columns[manager_col])))
    line_names = data.columns
    plt.figlegend(ax0.lines, line_names, loc=4)

    # axis titles
    ax0.set_ylabel("Cumulative Return")
    ax1.set_ylabel("Period Return")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")

    # format the yaxis
    ax_r = ax0.get_yticks()
    ax_v = ax1.get_yticks()
    ax_s = ax2.get_yticks()
    ax0.set_yticklabels(['{:0.1f}%'.format(x * 100) for x in ax_r])
    ax1.set_yticklabels(['{:0.1f}%'.format(x * 100) for x in ax_v])
    ax2.set_yticklabels(['{:0.1f}%'.format(x * 100) for x in ax_s])

    return plt


def dd(series):
    cr = ((1 + series).cumprod())
    rm = cr.expanding(min_periods=1).max()
    ddown = (cr / rm) - 1
    return ddown
