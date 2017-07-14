# MIT License

# Copyright (c) 2017 Jacob Bourne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from performanceanalytics import statistics

def test_geomean_simple():
    assert statistics.geo_mean([2,18])==6
    assert statistics.geo_mean([10,51.2,8]) == pytest.approx(16,.000001)
    assert statistics.geo_mean([1,3,9,27,81]) == pytest.approx(9,.000001)


def test_geomean_timeseries(series):
    dv = series[series.columns[0]].values
    dv2 = series[series.columns[1]].values
    assert statistics.geo_mean_return(dv) == pytest.approx(0.0108,.001)
    assert statistics.geo_mean_return(dv2) == pytest.approx(0.0135, .001)


