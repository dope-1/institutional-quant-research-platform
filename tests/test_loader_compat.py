import sys
import types

import pandas as pd

try:
    import yfinance  # noqa: F401
except ImportError:
    sys.modules["yfinance"] = types.SimpleNamespace(download=None)

try:
    import pandas_datareader  # noqa: F401
except ImportError:
    fake_data = types.SimpleNamespace(DataReader=None, get_data_fred=None)
    fake_package = types.ModuleType("pandas_datareader")
    fake_package.data = fake_data
    sys.modules["pandas_datareader"] = fake_package

from quant_platform.data import loader


def test_fama_french_period_index_is_converted(monkeypatch):
    idx = pd.period_range("2024-01-02", periods=2, freq="D")
    table = pd.DataFrame({"Mkt-RF": [1.0, 2.0], "RF": [0.1, 0.1]}, index=idx)

    def fake_reader(*args, **kwargs):
        return {0: table}

    monkeypatch.setattr(loader.pdr, "DataReader", fake_reader)
    out = loader.download_fama_french_daily("dummy", "2024-01-01", "2024-01-31")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.dtype) == "datetime64[ns]"
    assert out.loc[pd.Timestamp("2024-01-02"), "mkt_rf"] == 0.01
