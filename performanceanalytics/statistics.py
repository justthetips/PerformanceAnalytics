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


def create_capm_frame(manager,index,rf=None):
    # lets check to make sure manager and index are pandas series
    if not isinstance(manager,pd.Series):
        raise ValueError("Manager series must be a pandas series")
    if not isinstance(index,pd.Series):
        raise ValueError("Index series must be a pandas series")
    if (rf is not None) and (not isinstance(rf,pd.Series)):
        raise ValueError("Risk Free must either be none or a pandas series")

    #check for lengths, we do this befor the na's
    if manager.size != index.size:
        raise ValueError("Manager and Index must be the same size, you passed in {} and {}".format(manager.size,index.size))
    if (rf is not None) and (manager.size != rf.size):
        raise ValueError("Manager and RF must be the same size, you passed in {} and {}".format(manager.size, index.size))



    #if the risk free is None, create a risk free series of 0
    if rf is None:
        rf_data = [0.0] * len(manager)
        rf = pd.Series(rf_data,index=manager.index)


    #drop the na's and join to make sure they have the same valid length
    manager = manager.dropna()
    index = index.dropna()
    rf = rf.dropna()
    df = pd.concat([manager,index,rf],axis=1,join='inner')

    #return the df
    return df

def capm(manager,index,rf=None):
    df = create_capm_frame(manager,index,rf)

    #now that we have the dataframe, we subtract the rf from the manager and the index
    manager_adj = df[df.columns[0]] - df[df.columns[2]]
    index_adj = df[df.columns[1]] - df[df.columns[2]]

    #use numpy linalg to calculate the terms
    x = index_adj.values
    y = manager_adj.values
    A = np.vstack([x, np.ones(len(x))]).T

    m, c = np.linalg.lstsq(A, y)[0]

    r2 = np.corrcoef(x,y)[0][1] ** 2

    #return as a tuple
    return (c, m, r2)






