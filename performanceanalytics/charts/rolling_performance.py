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

def create_rollingperformance(data,width,rF=0,annual_factor=12.0,*args,**kwargs):
    # create the rolling vectors
    r  = data.rolling(window=width,center=False).apply(r_return,args=[width/annual_factor])
    v = data.rolling(window=width,center=False).apply(r_stdev,args=[annual_factor])
    s = (r-rF) / v

    #create the plots
    f, axarr = plt.subplots(3,sharex=True)
    axarr[0].plot(r)
    axarr[1].plot(v)
    axarr[2].plot(s)

    f.set_size_inches(kwargs.pop('figsize',(8,6)))

    # title and legend
    f.suptitle(kwargs.pop('title','Rolling Performance Summary'))
    line_names = data.columns
    plt.figlegend(axarr[0].lines,line_names,loc=4)

    #axis titles
    axarr[0].set_ylabel("Annualized Return")
    axarr[1].set_ylabel("Annualzied Vol")
    axarr[2].set_ylabel("Sharpe Ratio")
    axarr[2].set_xlabel("Date")

    #format the yaxis
    ax_r = axarr[0].get_yticks()
    ax_v = axarr[1].get_yticks()
    ax_s = axarr[2].get_yticks()
    axarr[0].set_yticklabels(['{:0.1f}%'.format(x*100) for x in ax_r])
    axarr[1].set_yticklabels(['{:0.1f}%'.format(x * 100) for x in ax_v])
    axarr[2].set_yticklabels(['{:0.2f}'.format(x) for x in ax_s])

    return plt




def r_return(data,factor):
    total_return = ((1 + data).cumprod())[-1]
    ar = (total_return ** (1/factor))-1
    return ar

def r_stdev(data,factor):
    return np.std(data) * np.sqrt(factor)


