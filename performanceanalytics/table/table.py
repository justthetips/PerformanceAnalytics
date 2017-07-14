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
import os
from collections import namedtuple
from performanceanalytics.statistics import geo_mean_return, mean_confidence_interval


def calendar_returns(data_series, manager_col=0, index_cols=None, as_perc=False):
    """
    creates a table of monthly and calendar year returns.  Please note, if there is no data
    for the index in the year it will go in as 0, this might be misleading, but it is better than errors for
    non complete years
    :param data_series: the data series
    :param manager_col: the column that has the manager in it (defaults to 0)
    :param index_cols: the column(s) that have the indecies in it (defaults to none)
    :param as_perc: if true, all values will be multiplied by 100
    :return: a data frame
    """

    # the series for the manager
    _s = data_series[data_series.columns[manager_col]].dropna()

    # get the range of years
    first_year = _s.index.min().year
    last_year = _s.index.max().year

    # create the column and row labels
    years = list(range(first_year, last_year + 1))
    months = list(range(1, 13))

    # create a data frame with the years as the columns and the months as the rows
    df = pd.DataFrame(0, months, years, float)

    # loop thru the time series and build the dataframe
    for yc, y in enumerate(years):
        for mc, m in enumerate(months):
            # turn y amd m into a date string
            idx = '-'.join([str(y), str(m)])
            # see if the index is in there
            if idx in _s.index:
                df.iloc[mc][y] = _s[idx]

    # we now have the data frame, now we can append the annual returns as the last row
    annual_returns = []
    for y in years:
        annual_returns.append(((df[y] + 1).cumprod() - 1).iloc[11])
    # append it to the bottom
    df.loc[data_series.columns.values[manager_col]] = annual_returns

    # now we have to add index stuff if any to the bottom of the table
    if index_cols is not None:
        for index_col in index_cols:
            _i = data_series[data_series.columns[index_col]].fillna(0)
            index_annual_returns = []
            for y in years:
                index_annual_returns.append(((_i[str(y)] + 1).cumprod() - 1).iloc[11])
            df.loc[data_series.columns.values[index_col]] = index_annual_returns

    if as_perc:
        df = df.applymap(lambda x: "{0:.2f}%".format(x * 100))

    return df


def stats_table(data_series, manager_col=0, other_cols=None):
    """
    create the stats table.  the stats table has summary statistical data of all the time series passed in
    :param data_series: the data series
    :param manager_col: the column for the manager, defaults to 0, will always be the first column
    :param other_cols: an optional list of other series to run stats on
    :return: the table
    """
    manager_stats = series_stats(data_series[data_series.columns[manager_col]])
    other_stats = []
    # check to see if there are other series to run on
    if other_cols is not None:
        other_stats = [series_stats(data_series[data_series.columns[x]]) for x in other_cols]
    # row names
    rows = ['Observations', 'NAs', 'Minimum', 'Quartile 1', 'Median', 'Artithmetic Mean', 'Geometric Mean',
            'Quartile 3', 'Maximum', 'SE Mean', 'LCL Mean (.95)', 'UCL Mean (.95)', 'Variance', 'Stdev', 'Skewness',
            'Kurtosis']
    # pandas is great, but renaming columns is a pain, this creates a list of column names
    cols = data_series.columns.tolist()
    colnames = [cols[manager_col]]
    if other_cols is not None:
        for c in other_cols:
            colnames.append(cols[c])

    # create a list of stats objects
    st = [manager_stats]
    # append other stats if any
    for os in other_stats:
        st.append(os)
    # create a dictionary to build the dataframe
    st_data = {}
    st_data['Observations'] = [x.Observations for x in st]
    st_data['NAs'] = [x.NAs for x in st]
    st_data['Minimum'] = [x.Minimum for x in st]
    st_data['Quartile 1'] = [x.Quartile1 for x in st]
    st_data['Median'] = [x.Median for x in st]
    st_data['Artithmetic Mean'] = [x.aMean for x in st]
    st_data['Geometric Mean'] = [x.gMean for x in st]
    st_data['Quartile 3'] = [x.Quartile3 for x in st]
    st_data['Maximum'] = [x.Maximum for x in st]
    st_data['SE Mean'] = [x.seMean for x in st]
    st_data['LCL Mean (.95)'] = [x.lclMean for x in st]
    st_data['UCL Mean (.95)'] = [x.uclMean for x in st]
    st_data['Variance'] = [x.Variance for x in st]
    st_data['Stdev'] = [x.Stdev for x in st]
    st_data['Skewness'] = [x.Skew for x in st]
    st_data['Kurtosis'] = [x.Kurt for x in st]

    df = pd.DataFrame.from_dict(st_data,orient='index')
    #rename the columns
    col_name_dict = {}
    for counter,old_name in enumerate(df.columns.tolist()):
        col_name_dict[old_name] = colnames[counter]
    df.rename(columns=col_name_dict,inplace=True)

    return df


def series_stats(data_series):
    """
    takes a single panda series and returns a named tuple with all the stats for the stats table
    :param data_series: the single pandas series
    :return: a named tuple with all the values
    """
    if not isinstance(data_series, pd.Series):
        raise ValueError("Must be a Pandas Series")

    SContainer = namedtuple('SContainer',
                            'Observations NAs Minimum Quartile1 Median aMean gMean Quartile3 Maximum seMean lclMean uclMean Variance Stdev Skew Kurt')

    SContainer.Observations = data_series.count()
    SContainer.NAs = data_series.isnull().sum()
    SContainer.Minimum = data_series.min()
    SContainer.Quartile1 = data_series.quantile(.25)
    SContainer.Median = data_series.median()
    SContainer.gMean = geo_mean_return(data_series)
    SContainer.Quartile3 = data_series.quantile(.75)
    SContainer.Maximum = data_series.max()
    SContainer.seMean = data_series.sem()
    SContainer.aMean, SContainer.lclMean, SContainer.uclMean = mean_confidence_interval(data_series)
    SContainer.Variance = data_series.var()
    SContainer.Stdev = data_series.std()
    SContainer.Skew = data_series.skew()
    SContainer.Kurt = data_series.kurt()
    return SContainer


base_path = os.path.abspath(os.getcwd())
data_file = os.path.join(base_path, 'data', 'managers.csv')
series = pd.read_csv(data_file, index_col=0, parse_dates=[0])
stats_table(series,manager_col=0,other_cols=[1,2,3,4])
