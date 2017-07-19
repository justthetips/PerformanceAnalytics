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


def find_drawdowns(series):
    """
    find the drawdowns of a series, returns a list of drawdown holder objects
    :param series: the series
    :return: list of drawdown holders
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Only works for Pandas Series, you passed in {}".format(type(series)))

    # first turn the series into the cumprod
    dd_series = (1 + series).cumprod()

    # now walk through the time series finding the dd
    prior_max = dd_series.iloc[0]
    prior_min = prior_max
    in_drawdown = False
    current_dd = None
    dd_list = []

    for dt, value in dd_series.iteritems():
        # if the value is lower than the previous we are in a drawdown
        if value < prior_max:
            # if we are not already in a drawdown we are now
            if not in_drawdown:
                in_drawdown = True
                dd = DrawdownHolder(dt)
                dd.max_value = prior_max
                dd.min_value = value
                dd.trough_date = dt
                prior_min = value
                current_dd = dd

            elif value < prior_min:
                # if we are in a drawdown, check to see if we are at the min
                current_dd.min_value = value
                current_dd.trough_date = dt
                prior_min = value
        else:
            if in_drawdown:
                # the drawdown is over
                current_dd.end_date = dt
                prior_max = value
                in_drawdown = False
                dd_list.append(current_dd)
            else:
                prior_max = value

    return dd_list


class DrawdownHolder(object):
    """
    Custom class to hold all the information about a drawdown
    """

    def __init__(self, dd_start):
        """
        initialization, must pass in the start date
        :param dd_start:
        """
        self._dd_start = dd_start

    @property
    def start_date(self):
        """
        the start date
        :return: the start date of the drawdown
        """
        return self._dd_start

    @property
    def trough_date(self):
        """
        the date of the trough of the drawdown
        :return: the date
        """
        return self._trough_date

    @trough_date.setter
    def trough_date(self, td):
        """
        set the trough date
        :param td: the date
        :return:
        """
        self._trough_date = td

    @property
    def end_date(self):
        """
        the end date of the drawdown
        :return: the date
        """
        return self._end_date

    @end_date.setter
    def end_date(self, ed):
        """
        the end date of the drawdown
        :param ed: the date
        :return:
        """
        self._end_date = ed

    @property
    def max_value(self):
        """
        the max value before the drawdown began
        :return: the value
        """
        return self._max_value

    @max_value.setter
    def max_value(self, mv):
        """
        the max value before the drawdown began
        :param mv: the value
        :return:
        """
        self._max_value = mv

    @property
    def min_value(self):
        """
        the min value of the drawdown
        :return: the value
        """
        return self._min_value

    @min_value.setter
    def min_value(self, mv):
        """
        the min value of the drawdown
        :param mv: the value
        :return:
        """
        self._min_value = mv

    @property
    def depth(self):
        """
        the depth of the drawdown (min / max) - 1
        :return: the depth
        """
        if (self.min_value is None) or (self.max_value is None):
            raise AttributeError("Cannot be called until min value and max value are set")
        return (self.min_value / self.max_value) - 1

    @property
    def length(self):
        """
        the length of the drawdown in days
        :return: the length
        """
        if self.end_date is None:
            raise AttributeError("Cannot be called until the end date is set")
        return (self.end_date - self.start_date).days

    @property
    def recovery(self):
        """
        the length of the recovery in days
        :return: the length
        """
        if (self.trough_date is None) or (self.end_date is None):
            raise AttributeError("Cannot be called until trough date and end date are set")
        return (self.end_date - self.trough_date).days

    @property
    def to_trough(self):
        """
        the length from the start to the trough in days
        :return: the length
        """
        if self.trough_date is None:
            raise AttributeError("Cannot be called until trough date is set")
        return (self.trough_date - self.start_date).days

    def __repr__(self):
        return '{}: {} {} {}'.format(self.__class__.__name__,
                                     self.start_date,
                                     self.end_date, self.depth)

    def __lt__(self, other):
        return self.depth < other.depth

    def __le__(self, other):
        return self.depth <= other.depth

    def __gt__(self, other):
        return self.depth > other.depth

    def __ge__(self, other):
        return self.depth >= other.depth

    def __eq__(self, other):
        return self.start_date == other.start_date and self.trough_date == other.trough_date and self.end_date == other.end_date

    def __ne__(self, other):
        return self.start_date != other.start_date or self.trough_date == other.trough_date or self.end_date == other.end_date

def maxDrawDown(series):
    if not isinstance(series, pd.Series):
        raise ValueError("Only works for Pandas Series, you passed in {}".format(type(series)))
    cum_returns = (1 + series).cumprod()
    drawdown = 1 - cum_returns.div(cum_returns.cummax())
    return - drawdown.max()