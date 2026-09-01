from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from quant_platform.ml.evaluation import ic_summary, rank_ic_by_date

SUPPORTED_NONLINEAR_MODELS = ("random_forest", "xgboost", "lightgbm")


@dataclass(frozen=True)
class NonlinearSelectionResult:
    model_name: str
    selected_group: str
    selected_candidate: str
    selected_params: dict[str, Any]
    table: pd.DataFrame
    validation_predictions: pd.DataFrame


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


def build_nonlinear_model(
    model_name: str,
    params: dict[str, Any],
    *,
    random_seed: int,
):
    """Build one deterministic nonlinear regressor from a predeclared configuration."""
    name = str(model_name).lower()
    if name == "random_forest":
        defaults: dict[str, Any] = {
            "n_estimators": 400,
            "max_depth": 5,
            "min_samples_leaf": 25,
            "max_features": 0.75,
            "bootstrap": True,
            "n_jobs": -1,
            "random_state": int(random_seed),
        }
        defaults.update(params)
        defaults["random_state"] = int(random_seed)
        return RandomForestRegressor(**defaults)

    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "XGBoost is required for Milestone 2C. Install with pip install -e \".[dev,ml]\"."
            ) from exc
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 25.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 10.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": int(random_seed),
        }
        defaults.update(params)
        defaults["random_state"] = int(random_seed)
        return XGBRegressor(**defaults)

    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "LightGBM is required for Milestone 2C. Install with pip install -e \".[dev,ml]\"."
            ) from exc
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": -1,
            "min_child_samples": 75,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 10.0,
            "objective": "regression",
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": int(random_seed),
            "deterministic": True,
            "force_col_wise": True,
        }
        defaults.update(params)
        defaults["random_state"] = int(random_seed)
        return LGBMRegressor(**defaults)

    raise ValueError(
        f"Unknown nonlinear model '{model_name}'. Expected one of {SUPPORTED_NONLINEAR_MODELS}."
    )


def walk_forward_nonlinear_predictions(
    base_history: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    model_name: str,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, Any],
    candidate_name: str,
    min_training_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    """Expanding monthly predictions with the same strict label-maturity rule as Ridge."""
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
            feature_columns=feature_columns,
            target_column=target_column,
        )
        if len(training) < int(min_training_rows):
            raise ValueError(
                f"Only {len(training)} matured training rows available before {date.date()}, "
                f"below min_training_rows={min_training_rows}."
            )

        snapshot = eval_frame.loc[eval_frame["date"] == date].dropna(
            subset=feature_columns
        ).copy()
        if snapshot.empty:
            prior_eval = eval_frame.loc[eval_frame["date"] <= date].copy()
            continue

        model = build_nonlinear_model(model_name, params, random_seed=random_seed)
        model.fit(training[feature_columns], training[target_column])
        snapshot["prediction"] = model.predict(snapshot[feature_columns])
        snapshot["model_name"] = str(model_name)
        snapshot["candidate_name"] = str(candidate_name)
        snapshot["training_rows"] = int(len(training))
        snapshot["training_signal_end"] = pd.Timestamp(training["date"].max())
        snapshot["max_training_label_end"] = pd.Timestamp(training["label_end_date"].max())
        output.append(snapshot)
        prior_eval = eval_frame.loc[eval_frame["date"] <= date].copy()

    if not output:
        return eval_frame.assign(
            prediction=pd.Series(dtype=float),
            model_name=pd.Series(dtype=str),
            candidate_name=pd.Series(dtype=str),
            training_rows=pd.Series(dtype=int),
            training_signal_end=pd.Series(dtype="datetime64[ns]"),
            max_training_label_end=pd.Series(dtype="datetime64[ns]"),
        ).iloc[0:0]
    return pd.concat(output, ignore_index=True).sort_values(["date", "ticker"]).reset_index(
        drop=True
    )


def select_nonlinear_configuration(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    model_name: str,
    feature_groups: dict[str, list[str]],
    candidates: list[dict[str, Any]],
    target_column: str,
    actual_return_column: str,
    min_training_rows: int,
    min_assets_for_ic: int,
    random_seed: int,
) -> NonlinearSelectionResult:
    """Select feature family and model parameters using validation mean Rank IC only."""
    if not feature_groups:
        raise ValueError("feature_groups cannot be empty.")
    if not candidates:
        raise ValueError(f"No candidate configurations supplied for {model_name}.")

    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    params_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for group_name, features in feature_groups.items():
        for candidate_order, candidate in enumerate(candidates):
            if "name" not in candidate:
                raise ValueError(f"A {model_name} candidate is missing a 'name'.")
            candidate_name = str(candidate["name"])
            params = dict(candidate.get("params", {}))
            preds = walk_forward_nonlinear_predictions(
                train,
                validation,
                model_name=model_name,
                feature_columns=list(features),
                target_column=target_column,
                params=params,
                candidate_name=candidate_name,
                min_training_rows=min_training_rows,
                random_seed=random_seed,
            )
            ic = rank_ic_by_date(
                preds,
                actual_column=actual_return_column,
                min_assets=min_assets_for_ic,
            )
            summary = ic_summary(ic)
            rows.append(
                {
                    "model": model_name,
                    "feature_group": group_name,
                    "candidate": candidate_name,
                    "candidate_order": int(candidate_order),
                    "params_json": json.dumps(params, sort_keys=True),
                    **{f"validation_{k}": v for k, v in summary.items()},
                }
            )
            key = (group_name, candidate_name)
            predictions[key] = preds
            params_by_key[key] = params

    table = pd.DataFrame(rows).sort_values(
        ["validation_mean_rank_ic", "candidate_order", "feature_group", "candidate"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    valid = table.dropna(subset=["validation_mean_rank_ic"])
    if valid.empty:
        raise ValueError(f"No {model_name} candidate produced a valid validation Rank IC.")
    winner = valid.iloc[0]
    key = (str(winner["feature_group"]), str(winner["candidate"]))
    return NonlinearSelectionResult(
        model_name=str(model_name),
        selected_group=key[0],
        selected_candidate=key[1],
        selected_params=params_by_key[key],
        table=table,
        validation_predictions=predictions[key],
    )


def fit_reference_nonlinear(
    history: pd.DataFrame,
    *,
    before_date: pd.Timestamp,
    model_name: str,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, Any],
    random_seed: int,
):
    """Fit one pre-test reference nonlinear model without using test labels."""
    training = _prepare_training_rows(
        history,
        prediction_date=pd.Timestamp(before_date),
        feature_columns=feature_columns,
        target_column=target_column,
    )
    model = build_nonlinear_model(model_name, params, random_seed=random_seed)
    model.fit(training[feature_columns], training[target_column])
    return model, training


def native_feature_importance(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Return normalized native tree importance for transparent pre-test inspection."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")
    values = np.asarray(model.feature_importances_, dtype=float)
    if values.shape[0] != len(feature_columns):
        raise ValueError("Feature-importance length does not match feature columns.")
    total = float(values.sum())
    normalized = values / total if total > 0 else np.zeros_like(values)
    out = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values,
            "normalized_importance": normalized,
        }
    )
    return out.sort_values("normalized_importance", ascending=False).reset_index(drop=True)
