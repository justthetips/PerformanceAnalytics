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


def boxplot(data, manager_col=0, other_cols=None, **kwargs):
    """
    create a box and whisker chart.  There is not much here, but its a nice wrapper and
    keeps all the charing consitent
    :param data: the data
    :param manager_col: the manager column
    :param other_cols: any other columns to display
    :param kwargs: other arguments to pass to the plot
    :return: the plot
    """

    # prepare the data

    ax_cols = [manager_col]
    if other_cols is not None:
        for oc in other_cols:
            ax_cols.append(oc)

    df1 = data[data.columns[ax_cols]]
    df1 = df1.ix[::, ::-1]

    # box charts are so easy
    f = plt.figure(figsize=kwargs.pop('figsize', (8, 6)))
    ax = f.add_subplot(111)
    ax = df1.boxplot(grid=True, vert=False)

    # pretty it up a little bit
    f.suptitle(kwargs.pop('title', 'Return Distribution'))
    ax_t = ax.get_xticks()
    ax.set_xticklabels(['{:0.1f}%'.format(x * 100) for x in ax_t])

    return plt
