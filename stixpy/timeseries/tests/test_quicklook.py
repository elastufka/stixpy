from pathlib import Path

import pytest
from sunpy.timeseries import TimeSeries

from stixpy.data import test
from stixpy.timeseries.quicklook import *


def test_ql_lightcurve():
    ql_lc = TimeSeries(test.STIX_QL_LIGHTCURVE_TIMESERIES)
    assert isinstance(ql_lc, QLLightCurve)


def test_qlbackground():
    ql_lc = TimeSeries(test.STIX_QL_BACKGROUND_TIMESERIES)
    assert isinstance(ql_lc, QLBackground)


def test_qlvariance():
    ql_lc = TimeSeries(test.STIX_QL_VARIANCE_TIMESERIES)
    ql_lc.peek()
    assert isinstance(ql_lc, QLVariance)
