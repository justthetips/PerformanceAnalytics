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
import pandas as pa


class SeriesHolder(object):
    """
    This class holds time series data.  Strictly speaking it is not needed at all, it is just here
    for convienance sake.  Its an easy way to extract the manager, peer, index and rf rate time series
    for all the other perfromance attribution functions, you can just pass them in directly, so this is
    simply for convienance
    """

    def __init__(self, series, manager_col, rf_col, peer_cols=None, index_cols=None):
        """
        Initializer, if you are going to use it you must pass in the time seires and the manager
        column and the rf column
        :param series: a pandas time series
        :param manager_col: the column in the time series that represents the manager
        :param rf_col: the column in the time series that represents the risk free rate
        :param peer_cols: a list of columns that reperesent peers
        :param index_cols: a list of columns that represent indecies
        """
        self._series = series
        # make sure there is only one manager
        if not isinstance(manager_col, int):
            raise ValueError(
                "You can only specify a single column for managers, you specified {}".format(len(manager_col)))
        self._manager_col = manager_col
        # make sure there is only one risk free rate
        if not isinstance(rf_col, int):
            raise ValueError(
                "You can only specify a single column for the rf rate, you specified {}".format(len(rf_col)))
        self._rf_col = rf_col
        self._peer_cols = peer_cols
        self._index_cols = index_cols

    @property
    def series(self):
        """
        the pandas time series
        :return:  the time series
        """
        return self._series

    @property
    def manager_col(self):
        """
        the manager column
        :return: int for the column
        """
        return self._manager_col

    @property
    def rf_col(self):
        """
        the risk free rate column
        :return: int for the column
        """
        return self._rf_col

    @property
    def peer_cols(self):
        """
        the columns representing peers
        :return: a list of peer columns
        """
        return self._peer_cols

    @peer_cols.setter
    def peer_cols(self, peer_cols):
        """
        set the columns representing the peers, they default to none in the constructor
        :param peer_cols: list of ints
        :return:
        """
        # make sure there is at least one
        if len(peer_cols) < 1:
            raise ValueError("There must be at least on peer column, you specified {}".format(len(peer_cols)))
        self._peer_cols = peer_cols

    @property
    def index_cols(self):
        """
        the columns representing indecies
        :return: list of ints
        """
        return self._index_cols

    @index_cols.setter
    def index_cols(self, index_cols):
        """
        set the columns representing the indeciies, they default to none in the constrcutor
        :param index_cols: list of ints
        :return:
        """
        # make sure there is at least one
        if len(index_cols) < 1:
            raise ValueError("There must be at least one index column, you specifiec {}".format(len(index_cols)))
        self._index_cols = index_cols

    def get_manager_series(self):
        """
        get the time series for the manager
        :return: pandas time series
        """
        return self._series[self._manager_col]

    def get_peer_series(self):
        """
        get the time series for peers.  will return none if its not set
        :return: pandas time series
        """
        if self._peer_cols is None:
            return None
        else:
            return self._series[self._peer_cols]

    def get_index_series(self):
        """
        get the time series for the indecies.  Will return none if they are not set
        :return: pandas time series
        """
        if self._index_cols is None:
            return None
        else:
            return self._series[self._index_cols]

    def get_rf_series(self):
        """
        get the time series for the risk free rate
        :return: pandas time series
        """
        return self._series[self._rf_col]
