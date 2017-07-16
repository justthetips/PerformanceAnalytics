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
import performanceanalytics.statistics as pas
import os


def create_rollingperformance(data, width, rF=0, annual_factor=12.0, **kwargs):
    """
    create a three subplot chart of rolling return, rolling stdev and rolling sharpe
    :param data: the data as a DataFrame of the returns to plot
    :param width: the width of the rolling period
    :param rF: risk free rate for sharpe (defaults to 0)
    :param annual_factor: annualization factor (defaults to 12, monthly)
    :param kwargs: extra kwargs mainly for titles
    :return: the chart
    """
    # create the rolling vectors
    r = data.rolling(window=width, center=False).apply(r_return, args=[width / annual_factor])
    v = data.rolling(window=width, center=False).apply(r_stdev, args=[annual_factor])
    s = (r - rF) / v

    # create the plots
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(r)
    axarr[1].plot(v)
    axarr[2].plot(s)

    f.set_size_inches(kwargs.pop('figsize', (8, 6)))

    # title and legend
    f.suptitle(kwargs.pop('title', 'Rolling Performance Summary'))
    line_names = data.columns
    plt.figlegend(axarr[0].lines, line_names, loc=4)

    # axis titles
    axarr[0].set_ylabel("Annualized Return")
    axarr[1].set_ylabel("Annualzied Vol")
    axarr[2].set_ylabel("Sharpe Ratio")
    axarr[2].set_xlabel("Date")

    # format the yaxis
    ax_r = axarr[0].get_yticks()
    ax_v = axarr[1].get_yticks()
    ax_s = axarr[2].get_yticks()
    axarr[0].set_yticklabels(['{:0.1f}%'.format(x * 100) for x in ax_r])
    axarr[1].set_yticklabels(['{:0.1f}%'.format(x * 100) for x in ax_v])
    axarr[2].set_yticklabels(['{:0.2f}'.format(x) for x in ax_s])

    return plt


def r_return(data, factor):
    """
    rolling version of the total return
    the annualization factor is a bit odd, it is basically how many widths are in a year.
    so if the width is 6 months then the factor should be .5, etc
    :param data: the np array of data
    :param factor: annualization factor.
    :return: the annualized total return
    """
    total_return = ((1 + data).cumprod())[-1]
    ar = (total_return ** (1 / factor)) - 1
    return ar


def r_stdev(data, factor):
    """
    rolling standard deviation.  here the annualiztion factor is periods per year, ie if this in monthly,
    the factor is 12
    :param data: the data
    :param factor: the annualization factor
    :return: the annualized stdev
    """
    return np.std(data) * np.sqrt(factor)
