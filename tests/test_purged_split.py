import pandas as pd

from quant_platform.data.preprocessing import purged_time_split


def test_purged_split_keeps_forward_labels_inside_partition():
    dates = pd.bdate_range("2024-01-01", periods=12)
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * len(dates),
            "close": range(100, 100 + len(dates)),
        }
    )
    splits = purged_time_split(
        df,
        train_end=str(dates[4].date()),
        validation_end=str(dates[7].date()),
        test_end=str(dates[11].date()),
        horizon=2,
    )

    assert list(splits["train"]["date"]) == list(dates[:3])
    assert list(splits["validation"]["date"]) == [dates[5]]
    assert list(splits["test"]["date"]) == list(dates[8:10])
