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

from performanceanalytics.table.table import series_stats
import pytest

# test tolerance, yes calling it mine is a holdout from my bond trading days
MINE = .0001

def test_series_stats(series):
    st = series[series.columns[0]]
    sc = series_stats(st)
    assert sc.Observations == 132
    assert sc.NAs == 0
    assert sc.Minimum == pytest.approx(-.0944,abs=MINE)
    assert sc.Quartile1 == pytest.approx(0,abs=MINE)
    assert sc.Median == pytest.approx(0.01115,abs=MINE)
    assert sc.aMean == pytest.approx(0.0111,abs=MINE)
    assert sc.gMean == pytest.approx(0.0108,abs=MINE)
    assert sc.Quartile3 == pytest.approx(0.0248,abs=MINE)
    assert sc.Maximum == pytest.approx(0.0692,abs=MINE)
    assert sc.seMean == pytest.approx(0.00223,abs=MINE)
    assert sc.lclMean == pytest.approx(0.0067,abs=MINE)
    assert sc.uclMean == pytest.approx(0.0155,abs=MINE)
    assert sc.Variance == pytest.approx(0.00065,abs=MINE)
    assert sc.Stdev == pytest.approx(0.0256,abs=MINE)
    assert sc.Skew == pytest.approx(-0.66644,abs=MINE)
    assert sc.Kurt == pytest.approx(2.50042,abs=MINE)

def test_series_stats_nas(series):
    st = series[series.columns[1]]
    sc = series_stats(st)
    assert sc.Observations == 125
    assert sc.NAs == 7
    assert sc.Minimum == pytest.approx(-.0371,abs=MINE)
    assert sc.Quartile1 == pytest.approx(-0.0098,abs=MINE)
    assert sc.Median == pytest.approx(0.0082,abs=MINE)
    assert sc.aMean == pytest.approx(0.0141,abs=MINE)
    assert sc.gMean == pytest.approx(0.0135,abs=MINE)
    assert sc.Quartile3 == pytest.approx(0.0252,abs=MINE)
    assert sc.Maximum == pytest.approx(0.1556,abs=MINE)
    assert sc.seMean == pytest.approx(0.0033,abs=MINE)
    assert sc.lclMean == pytest.approx(0.0076,abs=MINE)
    assert sc.uclMean == pytest.approx(0.0206,abs=MINE)
    assert sc.Variance == pytest.approx(0.0013,abs=MINE)
    assert sc.Stdev == pytest.approx(0.0367,abs=MINE)
    assert sc.Skew == pytest.approx(1.47581,abs=MINE)
    assert sc.Kurt == pytest.approx(2.52697,abs=MINE)

