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

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from functools import reduce


def geo_mean(data):
    """
    quickly calculate the geometric mean of a series
    :param data: the data to calc the mean
    :return: the geometric mean
    """
    return (reduce(lambda x, y: x * y, data)) ** (1.0 / len(data))


def geo_mean_return(data):
    """
    calculate the geometric mean return of a pandas time series.  please note
    na's are dropped so errors will not be returned
    :param data: the time series
    :return: the geomtreic mean return
    """
    data = data[~np.isnan(data)]
    cr = ((1 + data).cumprod())[-1]
    gr = (cr ** (1 / len(data))) - 1
    return gr


def annualized_return(data):
    total_return = ((1 + data).cumprod())[-1]
    s_date = data.index.min()
    e_date = data.index.max()
    t = ((e_date - s_date).days) / 365.25
    ar = (total_return ** (1 / t)) - 1
    return ar


def mean_confidence_interval(data, confidence=0.95):
    """
    calculate the mean and the upper and lower confidence bounds.  please note
    na's are dropped so errors will not be returned
    :param data: the data
    :param confidence: the confidence interval (defaults to .95)
    :return: tuple (mean, lcl, hcl)
    """
    data = data[~np.isnan(data)]
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def create_capm_frame(manager, index, rf=None):
    """
    create a suitable dataframe for CAPM calculations.  Series have to have the same length
    if the rf series is not passed in, one will be created with all 0s

    it is unlikely users will call this method, it is used internally

    :param manager: manager return series
    :param index: index return series
    :param rf: optional risk free rate series
    :return: a single data frame
    """
    # lets check to make sure manager and index are pandas series
    if not isinstance(manager, pd.Series):
        raise ValueError("Manager series must be a pandas series")
    if not isinstance(index, pd.Series):
        raise ValueError("Index series must be a pandas series")
    if (rf is not None) and (not isinstance(rf, pd.Series)):
        raise ValueError("Risk Free must either be none or a pandas series")

    # check for lengths, we do this befor the na's
    if manager.size != index.size:
        raise ValueError(
            "Manager and Index must be the same size, you passed in {} and {}".format(manager.size, index.size))
    if (rf is not None) and (manager.size != rf.size):
        raise ValueError(
            "Manager and RF must be the same size, you passed in {} and {}".format(manager.size, index.size))

    # if the risk free is None, create a risk free series of 0
    if rf is None:
        rf_data = [0.0] * len(manager)
        rf = pd.Series(rf_data, index=manager.index)

    # drop the na's and join to make sure they have the same valid length
    manager = manager.dropna()
    index = index.dropna()
    rf = rf.dropna()
    df = pd.concat([manager, index, rf], axis=1, join='inner')

    # return the df
    return df


def create_sharpe_frame(manager, rf):
    # lets check to make sure manager and index are pandas series
    if not isinstance(manager, pd.Series):
        raise ValueError("Manager series must be a pandas series")
    if not isinstance(rf, pd.Series):
        raise ValueError("Index series must be a pandas series")

    # check for lengths, we do this befor the na's
    if manager.size != rf.size:
        raise ValueError(
            "Manager and RF must be the same size, you passed in {} and {}".format(manager.size, rf.size))

    # drop the na's and join to make sure they have the same valid length
    manager = manager.dropna()
    rf = rf.dropna()
    df = pd.concat([manager, rf], axis=1, join='inner')

    # return the df
    return df


def capm(manager, index, rf=None):
    """
    calculate the CAPM parameters for the manager, and the index.  If the rf series is passed in, it will be subtracted from each
    :param manager: the manager time series
    :param index: the index time series
    :param rf: optional rf time series.
    :return: tuple of (alpha, beta, r2)
    """
    df = create_capm_frame(manager, index, rf)
    return capm_calc(df)


def capm_upper(manager, index, rf=None, threshold=0.0):
    """
    calculate CAPM on manager returns above a threshold (tehcnically above or equal to)
    :param manager: the manager
    :param index: the index time series
    :param rf: optional risk free rate time series
    :param threshold: the threshold, defaults to 0
    :return: tuple of (alpha, beta, r2)
    """
    df = create_capm_frame(manager, index, rf)
    df = df[df[df.columns[0]] >= threshold]
    return capm_calc(df)


def capm_lower(manager, index, rf=None, threshold=0.0):
    """
    calculate CAPM on manager returns below a threshold (tehcnically below or equal to)
    :param manager: the manager
    :param index: the index time series
    :param rf: optional risk free rate time series
    :param threshold: the threshold, defaults to 0
    :return: tuple of (alpha, beta, r2)
    """
    df = create_capm_frame(manager, index, rf)
    df = df[df[df.columns[0]] <= threshold]
    return capm_calc(df)


def capm_calc(df):
    """
    calculate alpha, beta, and r2 for a manager and an index series
    :param df: the capm dataframe
    :return: tuple (alpha, beta, r2)
    """
    # now that we have the dataframe, we subtract the rf from the manager and the index
    manager_adj = df[df.columns[0]] - df[df.columns[2]]
    index_adj = df[df.columns[1]] - df[df.columns[2]]

    # use numpy linalg to calculate the terms
    x = index_adj.values
    y = manager_adj.values
    A = np.vstack([x, np.ones(len(x))]).T

    m, c = np.linalg.lstsq(A, y)[0]

    r2 = np.corrcoef(x, y)[0][1] ** 2

    # return as a tuple
    return (c, m, r2)


def correl(manager, index, rf=None):
    """
    calculate to correlation and the correlation p-value between a manager and an index
    if the rf rate is passed in, they are both adjusted by it
    :param manager: the manager
    :param index: the index
    :param rf: rf defaults to None
    :return: tuple (correlation, pvalue)
    """
    df = create_capm_frame(manager, index, rf)
    return correl_calc(df)


def correl_calc(df):
    """
    internal calculation of the correlation, uses scypy
    :param df: the data frame
    :return: tuple (correlation,  pvalue)
    """
    # now that we have the dataframe, we subtract the rf from the manager and the index
    manager_adj = df[df.columns[0]] - df[df.columns[2]]
    index_adj = df[df.columns[1]] - df[df.columns[2]]

    # use numpy linalg to calculate the terms
    x = index_adj.values
    y = manager_adj.values

    # use scipy to calculate the correlation and the pvalue
    c, p = scipy.stats.pearsonr(x, y)
    return c, p


def tracking_error(manager, benchmark):
    """
    calculate the tracking error of a manager and the benchmark.
    Tracking error is calculated by taking the square root of the average of the squared deviations
    between the investment’s returns and the benchmark’s returns

    Since these calcs are all related, it returns a tuple of tracking error, active premium and information ratio
    :param manager: the manager
    :param benchmark: the benchmark
    :return: tuple (tracking error, active premium, information ratio)
    """
    df = create_capm_frame(manager, benchmark)
    m = df[df.columns[0]].values
    b = df[df.columns[1]].values
    diff = m - b
    te = np.sqrt(np.mean(diff ** 2))
    ap = np.mean(diff)
    ir = ap / np.std(diff)
    return (te, ap, ir)


def sharpe_ratio(manager, rF):
    df = create_sharpe_frame(manager, rF)
    m = df[df.columns[0]].values
    r = df[df.columns[1]].values
    sharpe = np.mean((m - r)) / np.std((m-r))
    return sharpe


def treynor_ratio(manager, index, rF):
    beta = capm(manager, index, rF)[1]
    df = create_sharpe_frame(manager, rF)
    m = df[df.columns[0]].values
    r = df[df.columns[1]].values
    treynor = np.mean((m - r)) / beta
    return treynor

def sortino_ratio(manager,rF):
    df = create_sharpe_frame(manager, rF)
    df = df[df[df.columns[0]] <= 0]
    m = df[df.columns[0]].values
    r = df[df.columns[1]].values
    sr = np.mean((m - r)) / np.std(m-r)
    return sr
