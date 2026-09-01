from __future__ import annotations

import pandas as pd


def equal_weight_scores(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index, name="score")


def momentum_scores(df: pd.DataFrame, feature: str = "momentum_12_1") -> pd.Series:
    if feature not in df:
        raise KeyError(f"Missing momentum feature: {feature}")
    return df[feature].rename("score")
