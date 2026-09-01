# Institutional Quant Research Platform

A leakage-aware, reproducible cross-sectional equity research platform built around the question:

> Can machine-learning models improve cross-sectional equity return ranking and risk-adjusted portfolio performance relative to conventional factor-style baselines after transaction costs?

The research engine is frozen after Milestone 3D. Milestone 4A adds a read-only
FastAPI product boundary and deterministic artifact-to-web JSON export. Milestone
4B adds a Next.js institutional dashboard over those exports, 4C packages both
services with Docker Compose, and 4D adds public-repository safeguards,
deployment configuration, and CI/CD. None of these milestones changes or
retrains the research methodology.

## Current status — Milestone 4D

Completed infrastructure includes:

- 60-name liquid US equity prototype universe
- Yahoo OHLCV adapter
- FRED macro adapter
- Kenneth French daily factor adapter
- OHLCV validation with numerical-tolerance handling for adjusted-price rounding
- momentum, volatility, and liquidity features
- 21-trading-observation forward-return labels
- backward-only macro as-of alignment
- purged train / validation / test splits
- equal-weight and momentum top-quintile baselines
- one-trading-day signal execution lag
- drift-aware portfolio accounting between monthly rebalances
- transaction costs and slippage
- Sharpe, Sortino, annualised return/volatility, max drawdown, VaR/CVaR, turnover, information ratio
- target-alignment and future-mutation leakage tests
- anomaly and coverage audit outputs
- production Docker and Docker Compose configuration
- GitHub Actions checks for Python, Next.js, deterministic exports, and images
- Vercel dashboard and Render API deployment configuration
- manual post-deployment smoke test workflow
- true calendar month-end ML sampling
- centered cross-sectional rank transforms for stock-specific features and target
- expanding Ridge walk-forward fitting with label-maturity checks
- validation-only alpha selection
- Spearman Rank IC and decile diagnostics
- test-period Ridge portfolio compared with equal-weight and momentum on identical signal dates

The static universe remains survivorship-biased and is an engineering/research prototype, not an investable alpha claim.

## Run with Python

Python 3.12 or 3.13 is recommended.

```bash
python -m venv .venv
# activate .venv
python -m pip install --upgrade pip
pip install -e ".[dev,ml]"
pytest
quant-research export-web
quant-research serve-api
```

The API is served at `http://127.0.0.1:8000`; interactive OpenAPI documentation
is at `http://127.0.0.1:8000/docs`. The exporter writes stable dashboard files to
`web/public/data/`.

### Milestone 4A endpoints

```text
GET /health
GET /api/v1/overview
GET /api/v1/models
GET /api/v1/performance
GET /api/v1/risk
GET /api/v1/rank-ic
GET /api/v1/explainability
GET /api/v1/regimes
GET /api/v1/optimization
GET /api/v1/holdout
GET /api/v1/methodology
```

The API and static exporter expose only whitelisted derived artifacts. They do
not retrain models, return raw OHLCV, publish online prediction rows, or expose
ticker-level optimizer targets. The negative early 2026 Rank IC is retained in
the public output.

## Run the Milestone 4B dashboard

Node.js 22.13 or newer is required. In a second terminal, leave the FastAPI
process running if you want to explore `/docs`, then start the dashboard:

```powershell
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. The dashboard reads `web/public/data/*.json`
directly, so FastAPI is supplemental and the UI still works when the API is not
running. After regenerating research summaries, run `quant-research export-web`
from the repository root and refresh the browser.

Production verification:

```powershell
cd web
npm run lint
npm run build
npm start
```

## Run both services with Docker — Milestone 4C

Install and start Docker Desktop, stop any manually running servers on ports
3000 and 8000, then run from the repository root:

```powershell
docker compose up --build
```

This starts the dashboard at `http://localhost:3000` and FastAPI at
`http://127.0.0.1:8000`. API documentation remains available at
`http://127.0.0.1:8000/docs`.

Background mode and shutdown:

```powershell
docker compose up --build -d
docker compose ps
docker compose down
```

The dashboard continues to use `web/public/data/*.json` as its primary source;
it does not block on the API. See `docs/milestone_4c_docker.md` for architecture,
refresh behavior, and container hardening details.

## Deploy with CI/CD — Milestone 4D

The deployment boundary remains static-JSON-first:

- GitHub Actions validates the Python package, deterministic exports, dashboard,
  and production containers.
- Vercel builds the Next.js project from repository root directory `web`.
- Render builds the supplemental API from the root `Dockerfile` using
  `render.yaml`.

The API image uses an explicit `QUANT_ARTIFACT_ROOT=/app/artifacts`; it does not
depend on editable installation paths. It also binds to Render's dynamic `PORT`
while retaining port 8000 for local Compose.

The exact account-connection and verification procedure is in
`docs/milestone_4d_deployment.md`. After both production URLs exist, run the
`deployment-smoke` workflow from the GitHub Actions tab and supply those two
base URLs.

### If Milestone 1.1 raw data already exists

Do **not** download everything again. Keep `data/raw/` and run:

```bash
pytest
quant-research run-milestone1 --reuse-raw
```

The command expects these existing snapshots:

```text
data/raw/market_ohlcv.parquet
data/raw/macro_fred.parquet
data/raw/fama_french_factors.parquet
```

It then rebuilds all processed data and reports with the Milestone 1.2 methodology.

## Outputs

```text
data/processed/
    research_panel.parquet
    split_summary.json
    splits/
        train.parquet
        validation.parquet
        test.parquet

artifacts/milestone1/
    performance_metrics.csv
    equity_curves.png
    equal_weight_daily.parquet
    momentum_daily.parquet
    audit/
        data_quality_summary.json
        ticker_coverage.csv
        cross_sectional_coverage.csv
        feature_diagnostics.csv
        annual_strategy_metrics.csv
        extreme_return_rows.csv
        zero_volume_rows.csv
        missing_macro_dates.csv
        cross_sectional_coverage.png
        annual_strategy_returns.png

data/processed/ml/
    train_monthly.parquet
    validation_monthly.parquet
    test_monthly.parquet
    dataset_summary.json

artifacts/milestone2a/
    ridge_model_selection.csv
    validation_predictions.parquet
    test_predictions.parquet
    validation_rank_ic.csv
    test_rank_ic.csv
    test_decile_forward_returns.csv
    ridge_coefficients_pretest.csv
    rank_ic_by_month.png
    test_decile_forward_returns.png
    ridge_coefficients.png
    test_equity_curves.png
    summary.json
    portfolio/
        performance_metrics.csv
        ridge_test_daily.parquet
        equal_weight_test_daily.parquet
        momentum_test_daily.parquet
```

## Why Milestone 1.2 exists

Real-data execution exposed several issues worth fixing before ML:

- adjusted OHLC data can differ by machine precision even when values are mathematically equal;
- the Fama/French reader can return a pandas `PeriodIndex`;
- newer pandas requires exact datetime precision matches for as-of merges;
- FRED series can contain missing observations on equity trading days;
- target-weight persistence without natural weight drift understates true rebalance turnover;
- a 21-day forward label near a split boundary can leak prices from the next partition.

Milestone 1.2 directly tests or corrects each of those issues. See `docs/milestone_1_2_hardening.md`.

## Research flow

```text
OHLCV + macro + factors
        ↓
validation / point-in-time alignment
        ↓
momentum / volatility / liquidity features
        ↓
21-day forward target
        ↓
purged chronological split
        ↓
equal-weight + momentum baselines
        ↓
drift-aware monthly portfolio simulation
        ↓
transaction costs + risk metrics
        ↓
research audit
        ↓
true month-end cross-sectional rank dataset
        ↓
validation-only Ridge alpha selection
        ↓
expanding walk-forward Ridge predictions
        ↓
Rank IC + decile analysis
        ↓
transaction-cost-aware test portfolio comparison
```

## Milestone 2A command

After Milestone 1.2 has completed successfully, run:

```bash
quant-research run-milestone2a
```

This command uses the existing processed panel and purged split files. It does **not** download data or rerun Milestone 1.2. See `docs/milestone_2a_ridge.md` for the methodology.

## Milestone 2B command

After Milestone 2A has completed successfully, run:

```bash
quant-research run-milestone2b
```

Milestone 2B performs validation-only feature ablation and portfolio selection, measures
walk-forward coefficient stability, adds bootstrap Rank-IC/spread diagnostics, and then
evaluates the frozen Ridge specification on the untouched test period. See
`docs/milestone_2b_robustness.md`.

## Next milestone

Milestone 4E can add a lightweight experiment registry containing experiment
IDs, git commits, dataset/configuration hashes, frozen model specifications, and
metrics. It must not change the research methodology or reinterpret the 2026
holdout.


### Milestone 2B outputs

```text
artifacts/milestone2b/
    feature_ablation_validation.csv
    selected_validation_predictions.parquet
    validation_coefficient_history.csv
    validation_coefficient_stability.csv
    portfolio_variant_validation.csv
    selected_test_predictions.parquet
    test_rank_ic.csv
    test_decile_forward_returns.csv
    test_top_bottom_spread.csv
    test_rank_ic_by_year.csv
    test_coefficient_history.csv
    coefficient_history.csv
    coefficient_stability.csv
    summary.json
    feature_ablation_validation_ic.png
    coefficient_stability.png
    validation_sharpe_vs_turnover.png
    test_rank_ic_by_year.png
    rank_ic_by_month.png
    test_decile_forward_returns.png
    test_equity_curves.png
    portfolio/
        performance_metrics.csv
        yearly_performance_metrics.csv
        ridge_selected_test_daily.parquet
        equal_weight_test_daily.parquet
        momentum_test_daily.parquet
```

## Milestone 2C — Nonlinear ML

Milestone 2C compares validation-selected Random Forest, XGBoost, and LightGBM models against the frozen Ridge benchmark, momentum, and equal weight using the same leakage-aware monthly research dataset and drift-aware transaction-cost backtester.

Run it after Milestone 2B:

```bash
quant-research run-milestone2c
```

Model feature groups, hyperparameters, and portfolio implementations are selected only on the validation period before the 2023–2025 evaluation is run. See `docs/milestone_2c_nonlinear.md` for the methodology and the disclosure that the historical test period has already been inspected in earlier milestones.

## Milestone 3A — explainability

After Milestone 2C, explain the frozen nonlinear specifications with SHAP, permutation importance, interaction diagnostics, and year-by-year stability:

```bash
pip install -e ".[dev,ml]"
quant-research run-milestone3a
```

Milestone 3A is post-hoc analysis only; it does not retune models on the historical evaluation period. Outputs are written to `artifacts/milestone3a/`.

## Milestone 3B — point-in-time market regime analysis

After Milestone 3A, diagnose whether the already-frozen ranking signals and portfolios behave differently across market environments:

```bash
quant-research run-milestone3b
```

Regimes include trailing-history VIX / Treasury / Fed-funds level states, 10Y rate direction, 10Y–2Y yield-curve shape, and trailing Fama–French market trend. Quantile thresholds are estimated from prior observations only. Milestone 3B is diagnostic and does not retune, switch, or resize any model or portfolio. See `docs/milestone_3b_regimes.md`.

## Milestone 3C — Institutional portfolio optimization

Milestone 3C keeps the Ridge/RF/XGBoost/LightGBM alpha models frozen and optimizes
portfolio construction instead. It uses point-in-time Ledoit-Wolf covariance,
naturally drifted pre-trade weights, long-only/full-investment constraints,
position caps, explicit L1 turnover penalties, and turnover caps. Optimizer
settings are selected on validation only.

Run:

```bash
quant-research run-milestone3c
```

See `docs/milestone_3c_portfolio_optimization.md` for methodology and disclosures.

## Milestone 3D — Genuine 2026 holdout

Run the frozen research methodology on incremental 2026 data without retuning:

```bash
quant-research run-milestone3d --as-of 2026-09-01
```

The command stores incremental data separately under `data/holdout2026/` and writes the prospective evaluation to `artifacts/milestone3d/`.


### As-of cutoff semantics
`--as-of YYYY-MM-DD` is an **exclusive** market-data cutoff. Daily OHLCV rows dated exactly on the as-of date are excluded, so an in-progress Yahoo session can never contaminate the prospective holdout. For `--as-of 2026-09-01`, the latest eligible completed daily bar is therefore 2026-08-31.
