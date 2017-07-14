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
    cr = ((1+data).cumprod())[-1]
    gr = (cr ** (1/len(data))) - 1
    return gr



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
