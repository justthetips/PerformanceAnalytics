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

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def create_histogram(data, manager_col=0, **kwargs):
    """
    create a chart of 4 histograms
    :param data: the data
    :param manager_col: the manager column (defaults to 0)
    :param kwargs: any charting stuff
    :return: the plot
    """
    # create the grid
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=kwargs.pop('figsize', (8, 6)))

    # we only use one column of data for all this
    df = data[data.columns[manager_col]]
    mu, sigma = stats.norm.fit(df)
    bins = 20

    ax1.hist(x=df, bins=bins)
    # we need the kde for the second chart
    n, b, p = ax2.hist(x=df, bins=bins)
    density = stats.gaussian_kde(df)
    kde_y = density(b)
    ax2.plot(b, kde_y, '--', linewidth=2)
    # now a normal line
    normal_y = mlab.normpdf(b, mu, sigma)
    ax2.plot(b, normal_y, '--', linewidth=2)
    # third chart
    ax3.hist(x=df, bins=bins)
    normal_y2 = mlab.normpdf(b, 0, sigma)
    ax3.plot(b, normal_y2, '--', linewidth=2)
    # fourth chart
    ax4.hist(x=df, bins=bins)
    var_line = np.percentile(df, 5)
    ax4.axvline(x=var_line, linestyle='--')
    ax4.text(x=var_line, y=ax4.get_yticks().max() * .9, s="VaR", rotation='vertical')

    # time to pretty things up
    ax_list = [ax1, ax2, ax3, ax4]
    for ax in ax_list:
        ax.set_xlabel("Returns")
        x_t = ax.get_xticks()
        ax.set_xticklabels(['{:0.1f}%'.format(x * 100) for x in x_t])
    ax1.set_ylabel("Freqency")
    ax2.set_ylabel("Density")
    ax3.set_ylabel("Freqency")
    ax4.set_ylabel("Density")

    f.suptitle(kwargs.pop('title', 'Return Distribution For {}'.format(data.columns[manager_col])))

    return plt
