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

# test tolerance, yes calling it mine is a holdout from my bond trading days
MINE = .0001

def test_geomean_simple():
    assert statistics.geo_mean([2,18])==6
    assert statistics.geo_mean([10,51.2,8]) == pytest.approx(16,abs=MINE)
    assert statistics.geo_mean([1,3,9,27,81]) == pytest.approx(9,abs=MINE)


def test_geomean_timeseries(series):
    dv = statistics.extract_returns_rf(series,0)[0]
    dv2 = statistics.extract_returns_rf(series,1)[0]
    assert statistics.geo_mean_return(dv) == pytest.approx(0.0108,abs=MINE)
    assert statistics.geo_mean_return(dv2) == pytest.approx(0.0135, abs=MINE)

def test_capm(series):
    manager, index, rf = statistics.extract_returns_bmark_rf(series,0,7,None,True)
    alpha, beta, r2 = statistics.capm(manager,index)
    assert alpha == pytest.approx(0.0077,abs=MINE)
    assert beta == pytest.approx(0.3906,abs=MINE)
    assert r2 == pytest.approx(0.4356,abs=MINE)

def test_capm_rf(series):
    manager, index, rf = statistics.extract_returns_bmark_rf(series, 0, 7, 9, True)
    alpha, beta, r2 = statistics.capm(manager,index,rf)
    assert alpha == pytest.approx(0.0057,abs=MINE)
    assert beta == pytest.approx(0.3900,abs=MINE)
    assert r2 == pytest.approx(0.4338,abs=MINE)

def test_capm_upper(series):
    manager, index, rf  = statistics.extract_returns_bmark_rf_partial(series,0,7,None,True,0,False)
    alpha, beta, r2 = statistics.capm(manager,index)
    assert alpha == pytest.approx(0.0177,abs=MINE)
    assert beta == pytest.approx(0.1984,abs=MINE)
    assert r2 == pytest.approx(0.1996,abs=MINE)

def test_capm_upper_rf(series):
    manager, index, rf = statistics.extract_returns_bmark_rf_partial(series, 0, 7, 9, True, 0, False)
    alpha, beta, r2 = statistics.capm(manager, index, rf)
    assert alpha == pytest.approx(0.0151,abs=MINE)
    assert beta == pytest.approx(0.1992,abs=MINE)
    assert r2 == pytest.approx(0.1985,abs=MINE)

def test_capm_lower(series):
    manager, index, rf  = statistics.extract_returns_bmark_rf_partial(series,0,7,None,True,0,True)
    alpha, beta, r2 = statistics.capm(manager,index)
    assert alpha == pytest.approx(-0.0100,abs=MINE)
    assert beta == pytest.approx(0.4000,abs=MINE)
    assert r2 == pytest.approx(0.6103,abs=MINE)

def test_capm_lower_rf(series):
    manager, index, rf = statistics.extract_returns_bmark_rf_partial(series, 0, 7, 9, True, 0, True)
    alpha, beta, r2 = statistics.capm(manager, index, rf)
    assert alpha == pytest.approx(-0.0119,abs=MINE)
    assert beta == pytest.approx(0.3991,abs=MINE)
    assert r2 == pytest.approx(0.6079,abs=MINE)

def test_capm_nas(series):
    manager, index, rf = statistics.extract_returns_bmark_rf(series,1,7,None,True)
    alpha, beta, r2 = statistics.capm(manager,index)
    assert alpha == pytest.approx(0.0111,abs=MINE)
    assert beta == pytest.approx(0.3431,abs=MINE)
    assert r2 == pytest.approx(0.1704,abs=MINE)

def test_correl(series):
    manager, index, rf = statistics.extract_returns_bmark_rf(series,0,7,None,True)
    c,p = statistics.correlation(manager,index)
    assert c == pytest.approx(0.6600,abs=MINE)
    assert p == pytest.approx(0,abs=MINE)

def test_te(series):
    manager, index, rf = statistics.extract_returns_bmark_rf(series, 0, 7, None, True)
    te = statistics.tracking_error(manager,index)
    ap = statistics.active_premium(manager,index)
    ir = statistics.information_ratio(manager,index)
    assert te == pytest.approx(0.3749,abs=MINE)
    assert ap == pytest.approx(0.0024, abs=MINE)
    assert ir == pytest.approx(0.0755, abs=MINE)

def test_annualized_return(series):
    manager, index, rf = statistics.extract_returns_bmark_rf(series, 0, 7, None, True)
    sd = series.index.min()
    ed = series.index.max()
    ar = statistics.annualized_return(manager,sd,ed)
    assert ar == pytest.approx(0.1386,abs=MINE)

def test_sharpelike(series):
    manager, index, rf = statistics.extract_returns_bmark_rf(series, 0, 7, 9, True)
    sharpe = statistics.sharpe_ratio(manager, rf)
    tr = statistics.treynor_ratio(manager,index,rf)
    sr = statistics.sortino_ratio(manager,rf)

    assert sharpe == pytest.approx(0.30927,abs=MINE)
    assert tr == pytest.approx(0.0202,abs=MINE)
    assert sr == pytest.approx(0.54304, abs=MINE)




