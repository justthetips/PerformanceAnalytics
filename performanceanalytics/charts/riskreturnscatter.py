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

import matplotlib.pyplot as plt
import numpy as np


def create_riskreturn_scatter(data, manager_col=0, other_cols=None, ann_factor=12, **kwargs):
    cols_used = [manager_col]
    for c, cols in enumerate(other_cols):
        cols_used.append(other_cols[c])
    # this data frame now has the columns we need
    df = data[data.columns[cols_used]]

    means = df.apply(np.mean)
    sigmas = df.apply(np.std)

    means = ((1 + means) ** ann_factor) - 1
    sigmas = sigmas * np.sqrt(ann_factor)

    # create the chart
    f = plt.figure(figsize=kwargs.pop('figsize', (8, 6)))
    ax = f.add_subplot(111)

    # create the plot
    ax.scatter(sigmas, means)
    ax.set_xlim([0, ax.get_xticks().max()])

    # add labels
    for cntr, lbl in enumerate(data.columns[cols_used]):
        x_pos = sigmas[cntr] * 1.01
        y_pos = means[cntr]
        ax.text(x_pos, y_pos, s=lbl)

    y_min = ax.get_yticks().min()
    y_max = ax.get_yticks().max()

    # now add some sharpe ratio lines
    sharpes = [0.5, 1, 1.5, 2.0]
    xs = ax.get_xticks()
    for sharpe in sharpes:
        ys = [x * sharpe for x in xs]
        ax.plot(xs, ys, linestyle='--', c='0.85')
    ax.set_ylim([y_min, y_max])

    # pretty some stuff up
    ax.set_xlabel("Annualized Risk")
    ax.set_ylabel("Annualized Return")
    ax.set_xticklabels(['{:0.1f}%'.format(x * 100) for x in ax.get_xticks()])
    ax.set_yticklabels(['{:0.1f}%'.format(x * 100) for x in ax.get_yticks()])

    f.suptitle(kwargs.pop('title', 'Risk vs Return'))

    return plt
