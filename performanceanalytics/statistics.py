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

# PLEASE NOTE, A lot of these functions are based on
# https://gist.github.com/StuartGordonReid/67a1ec4fbc8a84c0e856
# I would really like to thank him

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import numbers
import math
from functools import reduce


def vol(returns):
    """
    return the standard deviation of returns
    :param returns: returns
    :return: std
    """
    return np.std(returns)


def beta(returns, market):
    """
    calculate the beta between returns and the market
    :param returns: np.array of returns
    :param market: np.array of market
    :return: the beta
    """
    y = returns
    x = market
    # use numpy linalg to calculate the terms
    A = np.vstack([x, np.ones(len(x))]).T

    m, c = np.linalg.lstsq(A, y)[0]

    return m


def lpm(returns, threshold, order):
    """
    calculate the lower partial moment of returns
    :param returns: the returns
    :param threshold:  the threshold
    :param order: moment order
    :return: the lower partial moment
    """
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)


def hpm(returns, threshold, order):
    """
    calculate the higher partial moment of returns
    :param returns: the returns
    :param threshold:  the threshold
    :param order: moment order
    :return: the lower partial moment
    """
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the returns and the threshold
    diff = returns - threshold_array
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)


def var(returns, alpha):
    """
    calculate historical VaR
    :param returns: the returns
    :param alpha: the percentile
    :return: the VaR
    """
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])


def modified_var(returns, alpha):
    """
    calculate the modified VaR (for skew and kurtosis)
    :param returns: the returns
    :param alpha: the percnetile
    :return: the modified VaR
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    skew = scipy.stats.skew(returns)
    kurt = scipy.stats.kurtosis(returns)
    fp = scipy.stats.norm.cdf(alpha)

    z = fp + (fp ** 2 - 1) * skew / 6 + (fp ** 3 - 3 * fp) * kurt / 24 - (2 * fp ** 3 - 5 * fp) * skew ** 2 / 36

    return mu + (sigma * z)


def cvar(returns, alpha):
    """
    the conditional VaR
    :param returns: the returns
    :param alpha: the percentile
    :return: the cVaR
    """
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    # Return the average VaR
    # CVaR should be positive
    return abs(sum_var / index)


def prices(returns, base):
    """
    converts returns to an index for dds
    :param returns: returns
    :param base: base
    :return: array of index
    """
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return np.array(s)


def dd(returns, tau):
    """
    calculate the drawdown
    :param returns: the returns
    :param tau: the time period
    :return: array of vars
    """
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)


def max_dd(returns):
    """
    calculate the max drawdown
    :param returns: returns
    :return: the max dd
    """
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)


def average_dd(returns, periods):
    """
    the average drawdown over the period
    :param returns: the returns
    :param periods: the periods
    :return: the average drawdown
    """
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def average_dd_squared(returns, periods):
    """
    the average of the square of the drawdown over the periods
    :param returns: the returns
    :param periods: the periods
    :return: the average of the square of the dd
    """
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = math.pow(dd(returns, i), 2.0)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def treynor_ratio(returns, market, rf):
    """
    calculate the treynor ratio
    :param returns: returns
    :param market: market returns
    :param rf: risk free returns
    :return: the treynor ratio
    """
    return (np.average(returns - rf)) / beta(returns, market)


def sharpe_ratio(returns, rf):
    """
    calculate the sharpe ratio
    :param returns: the returns
    :param rf: the risk free rates
    :return: the sharpe ratio
    """
    return (np.average(returns - rf)) / vol(returns)


def information_ratio(returns, benchmark):
    """
    calculate the information ratio
    :param returns: the returns
    :param benchmark: the benchmark returns
    :return: the information ratio
    """
    diff = returns - benchmark
    return np.mean(diff) / vol(diff)


def tracking_error(returns, benchmark):
    """
    calculate the tracking error
    :param returns: the returns
    :param benchmark: the benchmark returns
    :return: the tracking error
    """
    diff = returns - benchmark
    return np.sqrt(np.sum(diff ** 2))


def active_premium(returns, benchmark):
    """
    calculate the active premium
    :param returns: the returns
    :param benchmark: the benchmark returns
    :return: the active premium
    """
    diff = returns - benchmark
    return np.mean(diff)


def modigliani_ratio(returns, benchmark, rf):
    """
    calculate the modigliani ratio
    :param returns: the returns
    :param benchmark: the benchmarke returns
    :param rf: the risk free rate
    :return: the modigliani ratio
    """
    np_rf = np.empty(len(returns))
    np_rf.fill(rf)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (np.average(returns - rf)) * (vol(rdiff) / vol(bdiff)) + rf


def excess_var(returns, rf, alpha):
    """
    calculate the excess var
    :param returns: the returns
    :param rf: the risk free rate
    :param alpha: the percentile
    :return: the excess var
    """
    return (np.average(returns - rf)) / var(returns, alpha)


def conditional_sharpe_ratio(returns, rf, alpha):
    """
    calculate the conditional sharpe ratio
    :param returns: the returns
    :param rf: the risk free rate
    :param alpha: the percentile
    :return: the conditional sharpe ratio
    """
    return (np.average(returns - rf)) / cvar(returns, alpha)


def omega_ratio(returns, rf, target=0):
    """
    calculate the omega ratio
    :param returns: the returns
    :param rf: the risk free rate
    :param target: the target return
    :return: the omega ratio
    """
    return (np.average(returns - rf)) / lpm(returns, target, 1)


def sortino_ratio(returns, rf, target=0):
    """
    calculate the sortino ratio
    :param returns: the returns
    :param rf: the risk free rate
    :param target: the target return
    :return: the sortino ratio
    """
    return (np.average(returns - rf)) / math.sqrt(lpm(returns, target, 2))


def kappa_three_ratio(returns, rf, target=0):
    """
    calculate the kappa three ratio
    :param returns: the returns
    :param rf: the risk free rate
    :param target: the return target
    :return: the ktr
    """
    return (np.average(returns - rf)) / math.pow(lpm(returns, target, 3), float(1 / 3))


def gain_loss_ratio(returns, target=0):
    """
    calculate the gain/loss ratio
    :param returns: the returns
    :param target: the target return
    :return: the gain/loss ratio
    """
    return hpm(returns, target, 1) / lpm(returns, target, 1)


def upside_potential_ratio(returns, target=0):
    """
    calculate the upside potential ratio
    :param returns: the returns
    :param target: the target return
    :return: the upr
    """
    return hpm(returns, target, 1) / math.sqrt(lpm(returns, target, 2))


def calmar_ratio(returns, rf):
    """
    calculate the calmar ratio
    :param returns: the returns
    :param rf: the risk free rate
    :return: the calmar ratio
    """
    return (np.average(returns - rf)) / max_dd(returns)


def sterling_ratio(returns, rf, periods):
    """
    calculare the sterling ratio
    :param returns: the returns
    :param rf: the risk free rate
    :param periods: the number of periods
    :return: the sterling ratio
    """
    return (np.average(returns - rf)) / average_dd(returns, periods)


def burke_ratio(returns, rf, periods):
    """
    calculate the burke ratio
    :param returns: the returns
    :param rf: the risk free rate
    :param periods: the number of periods
    :return: the burke ratio
    """
    return (np.average(returns - rf)) / math.sqrt(average_dd_squared(returns, periods))


def correlation(returns1, returns2, rtype='pearson'):
    """
    calculate the correlation between two return streems, can calculate any scipy correlation
    :param returns1: return 1
    :param returns2: return 2
    :param rtype: return type (by default pearson
    :return: the correlation and the pvalue (tuple)
    """
    cortype = ''.join([rtype, 'r'])
    c_func = getattr(scipy.stats, cortype)
    c, p = c_func(returns1, returns2)
    return c, p


def capm(returns, breturns, rfrates=None):
    """
    calculate the capm stats (alpha, beta, r2)
    :param returns: returns
    :param breturns: benchmark returns
    :param rfrates: riskfree rates (defaults to none)
    :return: tuple (alpha, beta, r2)
    """
    # if rfates are 0, just create a np.array or 0s
    if rfrates is None:
        rfrates = [0.0] * len(returns)
    # adj the returns
    y = returns - rfrates
    x = breturns - rfrates
    # use numpy linalg to calculate the terms
    A = np.vstack([x, np.ones(len(x))]).T

    m, c = np.linalg.lstsq(A, y)[0]

    r2 = np.corrcoef(x, y)[0][1] ** 2

    # return as a tuple
    return c, m, r2


def geo_mean(returns):
    """
    quickly calculate the geometric mean of a series
    :param returns: the data to calc the mean
    :return: the geometric mean
    """
    return (reduce(lambda x, y: x * y, returns)) ** (1.0 / len(returns))


def geo_mean_return(returns):
    """
    calculate the geometric mean return of a pandas time series.  please note
    na's are dropped so errors will not be returned
    :param returns: the time series
    :return: the geomtreic mean return
    """
    cr = ((1 + returns).cumprod())[-1]
    gr = (cr ** (1 / len(returns))) - 1
    return gr


def annualized_return(returns, start_date, end_date):
    """
    calculate the annualized geomteric return
    :param returns: the returns
    :param start_date: start date
    :param end_date: end date
    :return: the annualized return
    """
    total_return = ((1 + returns).cumprod())[-1]
    t = (end_date - start_date).days / 365.25
    ar = (total_return ** (1 / t)) - 1
    return ar


def mean_confidence_interval(returns, confidence=0.95):
    """
    calculate the mean and the upper and lower confidence bounds.
    :param returns: the data
    :param confidence: the confidence interval (defaults to .95)
    :return: tuple (mean, lcl, hcl)
    """
    a = 1.0 * np.array(returns)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    ci = scipy.stats.norm.interval(confidence, loc=m, scale=se / np.sqrt(n))
    return m, ci[0], ci[1]


def extract_returns_rf(dframe, return_col, rf_col=None, dropna=True, rf_default=0):
    """
    helper to extract the returns, and the rf rate of a cleaned series
    :param dframe: the data frame
    :param return_col: the return column
    :param rf_col: the risk free column (defaults to none, will use the rf_default if none)
    :param dropna: drop na's (true)
    :param rf_default: rf_default (0)
    :return: (returns, rfrates)
    """
    if not isinstance(dframe, pd.DataFrame):
        raise ValueError("dframe must be a pandas Dataframe, not a {}".format(type(dframe)))

    rc_name = dframe.columns[return_col]

    # if drop na remove rows where the manager has an n/a
    if dropna:
        dframe = dframe.dropna(subset=[rc_name])

    returns = dframe[rc_name].values
    # if a risk free col is set, extract that
    if rf_col is not None:
        rf = dframe[dframe.columns[rf_col]].values
    else:
        rf = [rf_default] * len(returns)

    return returns, rf


def extract_returns_bmark_rf(dframe, return_col, bmark_col, rf_col=None, dropna=True, rf_default=0):
    """

    helper to extract the returns, a benchmark return and the rf rate of a cleaned series
    :param dframe: the data frame
    :param return_col: the return column
    :param bmark_col: the benchmark column
    :param rf_col: the risk free column (defaults to none, will use the rf_default if none)
    :param dropna: drop na's (true)
    :param rf_default: rf_default (0)
    :return: (returns, bmark, rfrates)

    """
    if not isinstance(dframe, pd.DataFrame):
        raise ValueError("dframe must be a pandas Dataframe, not a {}".format(type(dframe)))

    rc_name = dframe.columns[return_col]
    bm_name = dframe.columns[bmark_col]

    # if drop na remove the rows where either the manager or the bmark has n/a
    if dropna:
        dframe = dframe.dropna(subset=[rc_name, bm_name], how='any')

    returns = dframe[rc_name].values
    bmark = dframe[bm_name].values

    if rf_col is not None:
        rf = dframe[dframe.columns[rf_col]].values
    else:
        rf = [rf_default] * len(returns)

    return returns, bmark, rf


def extract_returns_rf_partial(dframe, return_col, rf_col=None, dropna=True, threshold=0.0, lower=True, rf_default=0):
    """
    helper to extract the returns, and the rf rate of a cleaned series where the manager is above or below a threshold
    :param dframe: the data frame
    :param return_col: the return column
    :param rf_col: the risk free column (defaults to none, will use the rf_default if none)
    :param dropna: drop na's (true)
    :param threshold: the threshold (float or semi, if semi it will be the average of the returns)
    :param lower: True to return the lower half
    :param rf_default: the default risk free rate
    :return: (return, rf)
    """
    if not isinstance(dframe, pd.DataFrame):
        raise ValueError("dframe must be a pandas Dataframe, not a {}".format(type(dframe)))

    if not (isinstance(threshold, numbers.Real) or threshold.lower() == 'semi'):
        raise ValueError("threshold must be a float or semi, not {}".format(threshold))

    rc_name = dframe.columns[return_col]

    # if drop na remove rows where the manager has an n/a
    if dropna:
        dframe = dframe.dropna(subset=[rc_name])

    # calculare the alpha
    if isinstance(threshold, numbers.Real):
        alpha = threshold
    else:
        alpha = np.mean(dframe[rc_name].values)

    # trim the dataframe based on the threshold
    if lower:
        dframe = dframe.loc[lambda dframe: dframe[rc_name] <= alpha, :]
    else:
        dframe = dframe.loc[lambda dframe: dframe[rc_name] >= alpha, :]

        # since we already handled n/a we can just pass False
    return extract_returns_rf(dframe, return_col, rf_col, False, rf_default=rf_default)


def extract_returns_bmark_rf_partial(dframe, return_col, bmark_col, rf_col=None, dropna=True, threshold=0, lower=True, rf_default=0):
    """
    helper to extract the returns, and the rf rate of a cleaned series where the manager is above or below a threshold
    :param dframe: the data frame
    :param return_col: the return column
    :param bmark_col: the benchmark column
    :param rf_col: the risk free column (defaults to none, will use the rf_default if none)
    :param dropna: drop na's (true)
    :param threshold: the threshold (float or semi, if semi it will be the average of the returns)
    :param lower: True to return the lower half
    :param rf_default: the default risk free rate
    :return: (return, rf)
    """
    if not isinstance(dframe, pd.DataFrame):
        raise ValueError("dframe must be a pandas Dataframe, not a {}".format(type(dframe)))

    if not (isinstance(threshold, numbers.Real) or threshold.lower() == 'semi'):
        raise ValueError("threshold must be a float or semi, not {}".format(threshold))

    rc_name = dframe.columns[return_col]
    bm_name = dframe.columns[bmark_col]

    # if drop na remove the rows where either the manager or the bmark has n/a
    if dropna:
        dframe = dframe.dropna(subset=[rc_name, bm_name], how='any')

    # calculare the alpha
    if isinstance(threshold, numbers.Real):
        alpha = float(threshold)
    else:
        alpha = np.mean(dframe[rc_name].values)

    # trim the dataframe based on the threshold
    if lower:
        dframe = dframe.loc[lambda dframe: dframe[rc_name] <= alpha, :]
    else:
        dframe = dframe.loc[lambda dframe: dframe[rc_name] >= alpha, :]

    # since we already handled n/a we can just pass False
    return extract_returns_bmark_rf(dframe, return_col, bmark_col, rf_col, False, rf_default=rf_default)
