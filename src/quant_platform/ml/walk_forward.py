from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.ml.evaluation import ic_summary, rank_ic_by_date
from quant_platform.models.linear import build_ridge_model


@dataclass(frozen=True)
class AlphaSelectionResult:
    best_alpha: float
    table: pd.DataFrame
    predictions: pd.DataFrame


def _prepare_training_rows(
    history: pd.DataFrame,
    *,
    prediction_date: pd.Timestamp,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    eligible = history.loc[
        pd.to_datetime(history["label_end_date"]) < pd.Timestamp(prediction_date)
    ].copy()
    required = [target_column, *feature_columns]
    eligible = eligible.dropna(subset=required)
    numeric = eligible[required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric).all(axis=1)
    return eligible.loc[finite]


def walk_forward_ridge_predictions(
    base_history: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str = "target_rank",
    alpha: float = 1.0,
    min_training_rows: int = 500,
) -> pd.DataFrame:
    """Expanding monthly Ridge predictions with label-availability enforcement.

    For a signal at date ``t`` the fitted model may only use historical rows whose
    forward label has fully matured *before* ``t``. Earlier evaluation months join
    the expanding history only after their labels become observable.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    rank_features = list(feature_columns)
    history = base_history.copy()
    history["date"] = pd.to_datetime(history["date"]).astype("datetime64[ns]")
    history["label_end_date"] = pd.to_datetime(history["label_end_date"]).astype(
        "datetime64[ns]"
    )
    eval_frame = evaluation.copy()
    eval_frame["date"] = pd.to_datetime(eval_frame["date"]).astype("datetime64[ns]")
    eval_frame["label_end_date"] = pd.to_datetime(eval_frame["label_end_date"]).astype(
        "datetime64[ns]"
    )

    output: list[pd.DataFrame] = []
    prior_eval = eval_frame.iloc[0:0].copy()
    for date in sorted(eval_frame["date"].dropna().unique()):
        date = pd.Timestamp(date)
        expanding_history = pd.concat([history, prior_eval], ignore_index=True)
        training = _prepare_training_rows(
            expanding_history,
            prediction_date=date,
            feature_columns=rank_features,
            target_column=target_column,
        )
        if len(training) < min_training_rows:
            raise ValueError(
                f"Only {len(training)} matured training rows available before {date.date()}, "
                f"below min_training_rows={min_training_rows}."
            )

        snapshot = eval_frame.loc[eval_frame["date"] == date].dropna(
            subset=rank_features
        ).copy()
        if snapshot.empty:
            prior_eval = eval_frame.loc[eval_frame["date"] <= date].copy()
            continue

        model = build_ridge_model(alpha=alpha)
        model.fit(training[rank_features], training[target_column])
        snapshot["prediction"] = model.predict(snapshot[rank_features])
        snapshot["alpha"] = float(alpha)
        snapshot["training_rows"] = int(len(training))
        snapshot["training_signal_end"] = pd.Timestamp(training["date"].max())
        snapshot["max_training_label_end"] = pd.Timestamp(training["label_end_date"].max())
        output.append(snapshot)
        prior_eval = eval_frame.loc[eval_frame["date"] <= date].copy()

    if not output:
        return eval_frame.assign(
            prediction=pd.Series(dtype=float),
            alpha=pd.Series(dtype=float),
            training_rows=pd.Series(dtype=int),
            training_signal_end=pd.Series(dtype="datetime64[ns]"),
            max_training_label_end=pd.Series(dtype="datetime64[ns]"),
        ).iloc[0:0]
    return pd.concat(output, ignore_index=True).sort_values(["date", "ticker"]).reset_index(
        drop=True
    )


def select_ridge_alpha(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    actual_return_column: str,
    alpha_grid: list[float],
    min_training_rows: int,
    min_assets_for_ic: int,
) -> AlphaSelectionResult:
    """Choose Ridge alpha using validation walk-forward mean Rank IC only."""
    if not alpha_grid:
        raise ValueError("alpha_grid cannot be empty.")
    rows: list[dict[str, float]] = []
    predictions_by_alpha: dict[float, pd.DataFrame] = {}
    for alpha in alpha_grid:
        preds = walk_forward_ridge_predictions(
            train,
            validation,
            feature_columns=feature_columns,
            target_column=target_column,
            alpha=float(alpha),
            min_training_rows=min_training_rows,
        )
        ic = rank_ic_by_date(
            preds,
            prediction_column="prediction",
            actual_column=actual_return_column,
            min_assets=min_assets_for_ic,
        )
        summary = ic_summary(ic)
        rows.append({"alpha": float(alpha), **summary})
        predictions_by_alpha[float(alpha)] = preds

    table = pd.DataFrame(rows).sort_values(
        ["mean_rank_ic", "alpha"], ascending=[False, True], na_position="last"
    )
    valid = table.dropna(subset=["mean_rank_ic"])
    if valid.empty:
        raise ValueError("No alpha produced a valid validation Rank IC.")
    best_alpha = float(valid.iloc[0]["alpha"])
    return AlphaSelectionResult(
        best_alpha=best_alpha,
        table=table.reset_index(drop=True),
        predictions=predictions_by_alpha[best_alpha],
    )


def fit_reference_ridge(
    history: pd.DataFrame,
    *,
    before_date: pd.Timestamp,
    feature_columns: list[str],
    target_column: str,
    alpha: float,
):
    """Fit an interpretable pre-test reference model without using test labels."""
    training = _prepare_training_rows(
        history,
        prediction_date=pd.Timestamp(before_date),
        feature_columns=feature_columns,
        target_column=target_column,
    )
    model = build_ridge_model(alpha=alpha)
    model.fit(training[feature_columns], training[target_column])
    return model, training


def walk_forward_ridge_coefficients(
    base_history: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str = "target_rank",
    alpha: float = 1.0,
    min_training_rows: int = 500,
) -> pd.DataFrame:
    """Fit the same matured-label walk-forward models and record coefficients."""
    history = base_history.copy()
    history["date"] = pd.to_datetime(history["date"]).astype("datetime64[ns]")
    history["label_end_date"] = pd.to_datetime(history["label_end_date"]).astype("datetime64[ns]")
    eval_frame = evaluation.copy()
    eval_frame["date"] = pd.to_datetime(eval_frame["date"]).astype("datetime64[ns]")
    eval_frame["label_end_date"] = pd.to_datetime(eval_frame["label_end_date"]).astype("datetime64[ns]")

    rows: list[dict[str, object]] = []
    prior_eval = eval_frame.iloc[0:0].copy()
    for date in sorted(eval_frame["date"].dropna().unique()):
        date = pd.Timestamp(date)
        expanding_history = pd.concat([history, prior_eval], ignore_index=True)
        training = _prepare_training_rows(
            expanding_history,
            prediction_date=date,
            feature_columns=feature_columns,
            target_column=target_column,
        )
        if len(training) < min_training_rows:
            raise ValueError(
                f"Only {len(training)} matured training rows available before {date.date()}, "
                f"below min_training_rows={min_training_rows}."
            )
        model = build_ridge_model(alpha=alpha)
        model.fit(training[feature_columns], training[target_column])
        coefs = model.named_steps["model"].coef_
        for feature, coefficient in zip(feature_columns, coefs, strict=True):
            rows.append(
                {
                    "date": date,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "training_rows": int(len(training)),
                    "max_training_label_end": pd.Timestamp(training["label_end_date"].max()),
                }
            )
        prior_eval = eval_frame.loc[eval_frame["date"] <= date].copy()
    return pd.DataFrame(
        rows,
        columns=["date", "feature", "coefficient", "training_rows", "max_training_label_end"],
    )


def coefficient_stability_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Summarise sign consistency and dispersion of walk-forward Ridge coefficients."""
    if coefficients.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_coefficient",
                "median_coefficient",
                "std_coefficient",
                "positive_fraction",
                "negative_fraction",
                "dominant_sign_fraction",
                "months",
            ]
        )
    rows: list[dict[str, object]] = []
    for feature, group in coefficients.groupby("feature", sort=True):
        values = pd.to_numeric(group["coefficient"], errors="coerce").dropna()
        positive = float((values > 0).mean())
        negative = float((values < 0).mean())
        rows.append(
            {
                "feature": feature,
                "mean_coefficient": float(values.mean()),
                "median_coefficient": float(values.median()),
                "std_coefficient": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "positive_fraction": positive,
                "negative_fraction": negative,
                "dominant_sign_fraction": max(positive, negative),
                "months": int(len(values)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dominant_sign_fraction", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
