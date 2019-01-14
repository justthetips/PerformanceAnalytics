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


import collections

import numpy as np
import pandas as pd
from scipy.stats import sem, skew, kurtosis

import performanceanalytics.drawdowns as pad
import performanceanalytics.statistics as pas
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
    manager_stats = series_stats(data_series,manager_col)
    other_stats = []
    # check to see if there are other series to run on
    if other_cols is not None:
        other_stats = [series_stats(data_series,x) for x in other_cols]
    # row names
    # pandas is great, but renaming columns is a pain, this creates a list of column names
    cols = data_series.columns.tolist()
    colnames = [cols[manager_col]]
    if other_cols is not None:
        for c in other_cols:
            colnames.append(cols[c])

    # create a list of stats objects
    st = [manager_stats]
    # append other stats if any
    for ots in other_stats:
        st.append(ots)
    # create a dictionary to build the dataframe
    st_data = {'Observations': [x.Observations for x in st], 'NAs': [x.NAs for x in st],
               'Minimum': [x.Minimum for x in st], 'Quartile 1': [x.Quartile1 for x in st],
               'Median': [x.Median for x in st], 'Artithmetic Mean': [x.aMean for x in st],
               'Geometric Mean': [x.gMean for x in st], 'Quartile 3': [x.Quartile3 for x in st],
               'Maximum': [x.Maximum for x in st], 'SE Mean': [x.seMean for x in st],
               'LCL Mean (.95)': [x.lclMean for x in st], 'UCL Mean (.95)': [x.uclMean for x in st],
               'Variance': [x.Variance for x in st], 'Stdev': [x.Stdev for x in st], 'Skewness': [x.Skew for x in st],
               'Kurtosis': [x.Kurt for x in st]}

    df = pd.DataFrame.from_dict(st_data, orient='index')
    # rename the columns
    df = replace_col_names(df, colnames)

    return df


def series_stats(data, manager_col):
    """
    takes a single panda series and returns a named tuple with all the stats for the stats table
    :param data: the single pandas series
    :return: a named tuple with all the values
    """

    m_data = data[data.columns[manager_col]]
    m_data_clean = pas.extract_returns_rf(data, manager_col)[0]

    SContainer = collections.namedtuple('SContainer',
                                        'Observations NAs Minimum Quartile1 Median aMean gMean Quartile3 Maximum seMean lclMean uclMean Variance Stdev Skew Kurt')

    SContainer.Observations = len(m_data_clean)
    SContainer.NAs = m_data.isnull().sum()
    SContainer.Minimum = np.min(m_data_clean)
    SContainer.Quartile1 = np.percentile(m_data_clean, 25)
    SContainer.Median = np.median(m_data_clean)
    SContainer.gMean = geo_mean_return(m_data_clean)
    SContainer.Quartile3 = np.percentile(m_data_clean, 75)
    SContainer.Maximum = np.max(m_data_clean)
    SContainer.seMean = sem(m_data_clean)
    SContainer.aMean, SContainer.lclMean, SContainer.uclMean = mean_confidence_interval(m_data_clean)
    SContainer.Variance = np.var(m_data_clean)
    SContainer.Stdev = np.std(m_data_clean)
    SContainer.Skew = skew(m_data_clean)
    SContainer.Kurt = kurtosis(m_data_clean)
    return SContainer


def capm_table(data_series, manager_cols, index_col, rf_col):
    # check to make sure nothing is missing
    if index_col is None:
        raise ValueError("Index Column cannot be blank")
    if isinstance(index_col, collections.Iterable):
        raise ValueError("Index Column must be a single value")
    if manager_cols is None:
        raise ValueError("Manager column cannot be None")
    if not isinstance(manager_cols, collections.Iterable):
        manager_cols = [manager_cols]
    if rf_col is None:
        raise ValueError("Risk free column cannot be None")
    if isinstance(rf_col, collections.Iterable):
        raise ValueError("Risk free column must be a single value")

    # now create lists of the data points by comparing the manager to each comparison
    # TODO this is really ugly, fix
    alpha = []
    beta = []
    betap = []
    betam = []
    r2 = []
    r = []
    rp = []
    te = []
    ap = []
    ir = []
    tr = []
    for mc in manager_cols:
        manager, index, rf = pas.extract_returns_bmark_rf(data_series, mc, index_col, rf_col)
        manager_u, index_u, rf_u = pas.extract_returns_bmark_rf_partial(data_series, mc, index_col, rf_col, lower=False)
        manager_d, index_d, rf_d = pas.extract_returns_bmark_rf_partial(data_series, mc, index_col, rf_col)
        alpha.append(pas.capm(manager, index, rf)[0])
        beta.append(pas.capm(manager, index, rf)[1])
        r2.append(pas.capm(manager, index, rf)[2])
        betap.append(pas.capm(manager_u, index_u, rf_u)[1])
        betam.append(pas.capm(manager_d, index_d, rf_d)[1])
        r.append(pas.correlation(manager, index)[0])
        rp.append(pas.correlation(manager, index)[1])
        te.append(pas.tracking_error(manager, index))
        ap.append(pas.active_premium(manager, index))
        ir.append(pas.information_ratio(manager, index))
        tr.append(pas.treynor_ratio(manager, index, rf))
    st_data = {'Alpha': alpha,
               'Beta': beta,
               'Beta+': betap,
               'Beta-': betam,
               'R2': r2,
               'Correlation': r,
               'Correlation p-value': rp,
               'Tracking Error': te,
               'Active Premium': ap,
               'Information Ratio': ir,
               'Treynor Ratio': tr}

    df = pd.DataFrame.from_dict(st_data, orient='index')

    # replace the column names
    colnames = [' vs '.join([data_series.columns[x], data_series.columns[index_col]]) for x in manager_cols]
    # rename the columns
    df = replace_col_names(df, colnames)

    return df


def drawdown_table(data_series, trivial_dd=-0.02):
    """
    create a table of drawdowns.  the table shows the start, the trough, the end, the depth, the length, to trough
    and recovery.  those last three are in days
    :param data_series: the data series
    :param trivial_dd: drawdowns less than this are ignored, 2% by default
    :return: the table
    """
    drawdowns = pad.find_drawdowns(data_series)
    drawdowns = [x for x in drawdowns if x.depth <= trivial_dd]
    drawdowns = sorted(drawdowns)
    dd_list = []
    for dd in drawdowns:
        dd_list.append(dd_to_dict(dd))
    dd_frame = pd.DataFrame(dd_list)
    dd_frame = dd_frame[['From', 'Trough', 'End', 'Depth', 'Length', 'To Trough', 'Recovery']]
    return dd_frame


def dd_to_dict(dd):
    """
    turn the drawdown object into a dict
    :param dd: the dd object
    :return: a dictionary
    """
    ddict = {'From': dd.start_date, 'Trough': dd.trough_date, 'End': dd.end_date, 'Depth': dd.depth,
             'Length': dd.length,
             'To Trough': dd.to_trough, 'Recovery': dd.recovery}
    return ddict


def parse_cols(data, mc, ic, rfc):
    m = data[data.columns[mc]]
    i = data[data.columns[ic]]
    rf = data[data.columns[rfc]]
    return m, i, rf


def replace_col_names(df, colnames, inplace=True):
    if len(df.columns) != len(colnames):
        raise (
            "You must pass in the same number of column names as there are columns.  Column Size={}, Array Size={}".format(
                len(df.columns), len(colnames)))
    # create a dictionary to store position and name
    colname_dict = {}
    for counter, oldname in enumerate(df.columns.tolist()):
        colname_dict[oldname] = colnames[counter]
    df.rename(columns=colname_dict, inplace=inplace)
    return df


def create_downside_table(data, managercols, MAR=.02, rf=.005):
    # first see if managercols is a single
    if not isinstance(managercols, collections.Iterable):
        managercols = [managercols]

    colnames = []
    dstats = []

    for managercol in managercols:
        colnames.append(data.columns[managercol])
        dstats.append(downside_stats(data, managercol, MAR, rf))

    st_data = {'Semi Deviation': [x.semi for x in dstats],
               'Gain Deviation': [x.gain for x in dstats],
               'Loss Deviation': [x.loss for x in dstats],
               'Downside Deviation (MAR={0:.1f}%)'.format(MAR * 100): [x.ddmar for x in dstats],
               'Downside Deviation (rf={0:.1f}%)'.format(rf * 100): [x.ddrf for x in dstats],
               'Downside Deviation (0%)': [x.ddzero for x in dstats],
               'Maximum Drawdown': [x.mdd for x in dstats],
               'Historical VaR (95%)': [x.hvar for x in dstats],
               'Historical ES (95%)': [x.hes for x in dstats],
               'Modified VaR (95%)': [x.mvar for x in dstats],
               'Modified ES (95%)': [x.mes for x in dstats]
               }

    df = pd.DataFrame.from_dict(st_data, orient='index')
    # rename the columns
    df = replace_col_names(df, colnames)
    return df


def downside_stats(series, mcol, MAR, rf):
    if not isinstance(series, pd.DataFrame):
        raise ValueError("Must be a Pandas Series")

    dContainer = collections.namedtuple('dContainer', 'semi gain loss ddmar ddrf ddzero mdd hvar hes mvar mes')

    m, r = pas.extract_returns_rf(series, mcol, rf_col=None, rf_default=rf)
    m_mar, r_mar = pas.extract_returns_rf_partial(series, mcol, rf_col=None, rf_default=rf, threshold=MAR)
    m_rf, r_rf = pas.extract_returns_rf_partial(series, mcol, rf_col=None, rf_default=rf, threshold=rf)
    m_loss, r_loss = pas.extract_returns_rf_partial(series, mcol, rf_col=None, rf_default=rf, threshold=0)
    m_semi, r_semi = pas.extract_returns_rf_partial(series, mcol, rf_col=None, rf_default=rf, threshold='semi')
    m_gain, r_gain = pas.extract_returns_rf_partial(series, mcol, rf_col=None, rf_default=rf, threshold=0, lower=False)

    dContainer.semi = np.std(m_semi)
    dContainer.gain = np.std(m_gain)
    dContainer.loss = np.std(m_loss)
    dContainer.ddmar = np.std(m_mar)
    dContainer.ddrf = np.std(m_rf)
    dContainer.ddzero = np.std(m_loss)
    dContainer.mdd = min(pad.find_drawdowns(series))
    dContainer.hvar = pas.var(m, .05)
    dContainer.hes = pas.cvar(m, .05)
    dContainer.mvar = pas.modified_var(m, .05)
    dContainer.mes = pas.excess_var(m, r, .05)
    return dContainer
