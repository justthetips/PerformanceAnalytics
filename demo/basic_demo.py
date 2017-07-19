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


from performanceanalytics.charts.chartlib import default_plot
from performanceanalytics.charts.performance_summary import create_performance_summary
from performanceanalytics.charts.boxplot import boxplot
from performanceanalytics.charts.performance_histogram import create_histogram
from performanceanalytics.charts.riskreturnscatter import create_riskreturn_scatter
from performanceanalytics.charts.rolling_performance import create_rollingperformance
from performanceanalytics.charts.rolling_regression import create_rolling_regression
from performanceanalytics.table import table

import numpy as np
import matplotlib as plt
import pandas as pd
import os


def overview():
    # load the data
    base_path = os.path.abspath(os.path.dirname(__file__))
    data_file = os.path.join(base_path, os.pardir, 'data', 'managers.csv')
    rawdata = pd.read_csv(data_file, index_col=0, parse_dates=[0])

    # make sure the data is loaded
    print(rawdata.head(5))

    # create some helpful constants
    manager_col = 0
    peer_cols = [1, 2, 3, 4, 5]
    indexes_cols = [6, 7]
    rf_col = 9

    trailing_12 = rawdata.tail(12)
    trailing_36 = rawdata.tail(36)

    # get the default plot
    plt = default_plot()

    # create the performance summary chart
    create_performance_summary(rawdata, manager_col, peer_cols).show()

    # show the calendar returns
    cr_table = table.calendar_returns(rawdata, manager_col, indexes_cols)
    print(cr_table)

    # show the box plot
    boxplot(rawdata, manager_col, peer_cols + indexes_cols).show()

    # the histogram
    create_histogram(rawdata, manager_col).show()

    # risk returns scatter
    create_riskreturn_scatter(trailing_36,manager_col,peer_cols).show()

    # show stats
    st_table = table.stats_table(rawdata, manager_col, peer_cols)
    print(st_table)

    # rolling performance chart
    create_rollingperformance(rawdata[rawdata.columns[[manager_col]+peer_cols]],12).show()

    #CAPM table
    cp_table = table.capm_table(rawdata,[manager_col] + peer_cols,indexes_cols[1],rf_col)
    print(cp_table)

    # rolling regression
    create_rolling_regression(rawdata,12,indexes_cols[1],[manager_col] + peer_cols,rf_col).show()

    # downside table
    dtable = table.create_downside_table(rawdata,[manager_col] + peer_cols)
    print(dtable)


if __name__ == '__main__':
    overview()
