# Institutional Quant Research Platform - Complete Project Handoff

**Purpose of this file:** upload this file together with the current code folder into a new ChatGPT conversation. The next conversation should read this document before making any code changes. This is the continuity record for the frozen research engine through **Milestone 3D** and productization through **Milestone 4D (1 September 2026)**.

## 1. Immediate instruction to the next chat

Continue the existing repository **in place**. Do not redesign the research methodology, and do not retune the historical models because of the 2026 results. The research phase is frozen. The next task is:

> **Milestone 4E — add a lightweight experiment registry without changing the frozen methodology.**

Milestones 4A through 4D are complete in repository version 0.12.0. The 4D package contains CI/CD and provider configuration; connecting the user's GitHub, Vercel, and Render accounts remains an explicit user-controlled publishing step. The next code task is **Milestone 4E — experiment registry**. Preserve existing `.venv/`, `data/`, and `artifacts/` unless a task explicitly requires new outputs. New upgrades should continue the established drop-in pattern and must never package user raw data.

---

## 2. Project goal

Repository: `institutional-quant-research-platform`

The research question evolved into:

> Can machine-learning models improve cross-sectional equity ranking and risk-adjusted portfolio outcomes relative to conventional quantitative baselines after realistic chronology, transaction costs, model-selection discipline, explainability, regime diagnostics and institutional portfolio constraints are imposed?

This is deliberately **not** a stock-price prediction demo. It is a reproducible quantitative-research platform built around point-in-time feature construction, walk-forward model fitting, cross-sectional ranking, self-financing backtesting, statistical diagnostics, and a prospective holdout.

---

## 3. Current environment and package state

- Python requirement: `>=3.12,<3.15`
- Last known working Python: **3.12.11**
- Project package version: **0.12.0**
- Installation mode: editable (`pip install -e ".[dev,ml]"`)
- Main CLI: `quant-research`
- Dashboard runtime: Node.js `>=22.13.0`
- Dashboard framework: Next.js **16.3.4** (security-updated during 4D verification)
- Core libraries: pandas, numpy, pyarrow, scikit-learn, scipy, yfinance, pandas-datareader, matplotlib
- ML: XGBoost, LightGBM
- Explainability: SHAP
- Test suite: **70 passing tests** at the Milestone 4D handoff.

Current commands:

```powershell
quant-research run-milestone1 --reuse-raw
quant-research audit-milestone1
quant-research run-milestone2a
quant-research run-milestone2b
quant-research run-milestone2c
quant-research run-milestone3a
quant-research run-milestone3b
quant-research run-milestone3c
quant-research run-milestone3d --as-of 2026-09-01
```

---

## 4. Current repository architecture

```text
src/quant_platform/
├── cli.py
├── data/
│   ├── loader.py
│   ├── preprocessing.py
│   └── validation.py
├── features/
│   ├── momentum.py
│   ├── volatility.py
│   ├── liquidity.py
│   └── macro.py
├── models/
│   ├── baseline.py
│   ├── linear.py
│   └── tree_models.py
├── backtest/
│   ├── engine.py
│   └── walk_forward.py
├── portfolio/
│   ├── construction.py
│   ├── costs.py
│   └── risk.py
├── evaluation/
│   ├── audit.py
│   ├── audit_plots.py
│   ├── metrics.py
│   └── reports.py
└── ml/
    ├── walk_forward.py
    ├── portfolio.py
    ├── robustness.py
    ├── nonlinear.py
    ├── explainability.py
    ├── explainability_reports.py
    ├── regime.py
    ├── regime_reports.py
    ├── optimizer.py
    ├── optimizer_reports.py
    ├── holdout.py
    ├── holdout_reports.py
    └── reports.py
```

Test modules currently include validation, loader compatibility, features, no-leakage, purged splits, macro joins, backtesting, research audit, nonlinear ML, robustness, explainability, regimes, optimization and holdout integrity.

---

## 5. Data and research universe

### Historical period
- Frozen historical research range: **2012-01-03 to 2025-12-31**
- Universe: **60 liquid US equities**
- Historical validated market download: **211,200 rows / 60 tickers** before feature-panel filtering
- Research panel: **208,882 rows**
- Static universe is intentionally disclosed as survivorship-biased; it is an engineering/research prototype universe, not point-in-time index membership.

### Market data
- Prototype market source: Yahoo Finance through `yfinance`
- `auto_adjust: true`
- Raw snapshots are cached and should remain excluded from a public repository.

### Macro data
FRED series:
- `DGS10` -> `treasury_10y`
- `DGS2` -> `treasury_2y`
- `VIXCLS` -> `vix`
- `DFF` -> `fed_funds`

### Factor data
- Kenneth French daily `F-F_Research_Data_Factors_daily`

### 2026 prospective snapshot
- Separate incremental holdout data only; frozen 2012-2025 raw snapshot is never revised by 3D.
- Successful run as of 2026-09-01 used completed bars through **2026-08-31** for all 60 tickers.

---

## 6. Feature/label pipeline

Stock-specific ML features:

```text
momentum_21d
momentum_63d
momentum_126d
momentum_252d
momentum_6_1
momentum_12_1
volatility_21d
volatility_63d
log_dollar_volume
dollar_volume_median_21d
amihud_21d
```

Macro/factor values are retained for diagnostics/regimes but were not blindly included as same-day stock-ranking features because market-wide values have no within-date cross-sectional variation.

Target:

```text
forward_return_21d = P(t+21) / P(t) - 1
```

Every target also has a label-end date. A label may be used in training only when it is strictly matured before the current signal date.

Monthly ML uses centered cross-sectional percentile ranks for features and a centered cross-sectional rank of the 21-day forward return as the target.

---

## 7. Leakage and chronology safeguards

1. **Future-price mutation test:** modifying future prices cannot change historical features.
2. **Execution lag:** close-t signals only earn return starting on the next trading date.
3. **Self-financing weight drift:** portfolio weights drift with returns between rebalances; no implicit free daily rebalancing.
4. **Turnover measured against drifted pre-trade weights.**
5. **Purged partition boundaries:** a row remains in train/validation/test only if its 21-day label ends inside the same partition.
6. **Strict matured-label walk-forward:** `label_end_date < signal_date`.
7. **True calendar month-ends:** monthly samples are derived from the full trading calendar, not the last surviving row in a purged partition.
8. **Point-in-time macro joins:** backward-only carry of the last known macro observation; never future filling.
9. **Point-in-time regime thresholds:** trailing historical distributions only.
10. **Point-in-time covariance:** trailing returns on or before the signal date only.
11. **2026 as-of cutoff is exclusive:** incomplete current trading sessions are dropped before validation.

---

## 8. Purged splits

- Train: **117,142 rows**, 2012-01-03 to 2019-11-29
- Validation: **44,100 rows**, 2020-01-02 to 2022-11-30
- Historical evaluation: **43,860 rows**, 2023-01-03 to 2025-12-01
- Exactly **1,260 rows removed from each partition** by the 21-day purge (`60 × 21`).

Important terminology: 2023-2025 was initially untouched for 2A alpha selection, but after it was examined it became an **already-inspected historical evaluation period**. It must not be described as pristine after 2A. The only genuine prospective period is 2026.

---

## 9. Milestone chronology and results

### Milestone 1 / 1.1 / 1.2 - infrastructure and hardened baselines

Audit:
- research rows: **208,882**
- 60 tickers
- duplicate `(date,ticker)` rows: **0**
- label alignment mismatches: **0**
- non-finite feature values: **0**
- dates with all macro observations missing after hardening: **0**
- extreme >25% adjusted-return review flags: **11**
- zero/non-positive volume rows: **1**
- minimum assets/date: 56; median: 59
- audit: **PASS**

Hardened 2012-2025 baselines:
- Equal weight: annual return **20.25%**, vol **17.33%**, Sharpe **1.152**, max DD **-33.73%**, turnover **0.612x**
- Momentum top quintile: annual return **22.75%**, vol **21.86%**, Sharpe **1.048**, max DD **-32.27%**, turnover **5.773x**

### Milestone 2A - first Ridge experiment

Validation-selected alpha: `0.01`.
- Validation mean Rank IC: **0.0163**
- 2023-2025 mean Rank IC: **0.0371**
- top-minus-bottom decile forward-return spread: **1.61% per ~21 trading days**
- Ridge top-quintile annual return: **32.17%**
- Ridge Sharpe: **1.519**
- Ridge turnover: **13.50x**
- Equal-weight Sharpe over same window: **1.569**

This was encouraging but later robustness work showed the specification was not stable.

### Milestone 2B - linear robustness

Validation-only selection chose:
- feature group: **momentum_only**
- Ridge alpha: **10.0**
- portfolio: **top 10% equal weight**

Historical evaluation:
- mean Rank IC: **-0.0361**
- IC t-stat: **-0.93**
- 95% bootstrap CI: **[-0.1133, 0.0380]**
- positive IC months: **45.7%**
- top-bottom spread: **+0.51%**, t-stat **0.42**, CI spans zero
- portfolio annual return: **34.23%**, Sharpe **1.330**, turnover **19.60x**

Interpretation: high concentrated portfolio return did not imply robust full-cross-sectional ranking skill.

### Milestone 2C - nonlinear ML

Validation-only frozen specifications:

**Random Forest**
- feature group: all_features
- `n_estimators=400`, `max_depth=3`, `min_samples_leaf=40`, `max_features=0.75`
- portfolio: top20 rank-weighted
- validation IC **0.0412**
- historical IC **0.0341**, t-stat **1.13**, positive months **62.9%**
- annual return **25.80%**, Sharpe **1.157**, turnover **11.04x**

**XGBoost**
- all_features; conservative candidate
- `n_estimators=300`, `learning_rate=0.03`, `max_depth=2`, `min_child_weight=40`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=20`
- portfolio: top20 buffered (exit threshold 30%)
- validation IC **0.0370**
- historical IC **0.0148**, t-stat **0.49**
- annual return **27.60%**, Sharpe **1.367**, turnover **8.76x**

**LightGBM**
- all_features; shallow candidate
- `n_estimators=300`, `learning_rate=0.03`, `num_leaves=7`, `min_child_samples=75`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=10`
- portfolio: top20 buffered
- validation IC **0.0167**
- historical IC **0.0061**, t-stat **0.20**
- annual return **27.91%**, Sharpe **1.407**, turnover **10.44x**

Historical benchmark Sharpe remained best for equal-weight (**1.569**).

### Milestone 3A - explainability

3A is post-hoc only; it does not retune models.

Random Forest SHAP concentration:
- 21d median dollar volume: **30.65%** normalized |SHAP|
- log dollar volume: **16.21%**
- 21d momentum: **12.24%**
- 63d momentum: **8.91%**
- 6-1 momentum: **8.75%**

Top RF TreeSHAP interactions:
1. 21d momentum × 21d median dollar volume
2. 21d momentum × log dollar volume
3. 63d momentum × 21d median dollar volume
4. 63d volatility × 21d median dollar volume
5. 63d momentum × log dollar volume

Key interpretation: liquidity and volatility appear mainly as **conditioning variables** for momentum in nonlinear trees, explaining why tree models selected all_features while Ridge selected momentum only. SHAP is explanatory, not causal.

### Milestone 3B - point-in-time regime diagnostics

Diagnostic only; no regime switching or model retuning.

Regimes:
- VIX low/medium/high from trailing point-in-time terciles
- 10Y Treasury low/medium/high from trailing terciles
- Fed funds low/medium/high from trailing terciles
- 63-day rate direction: rising/flat/falling
- 10Y-2Y curve: normal/flat/inverted
- trailing 63d Fama-French market trend: positive/negative

Selected observations:
- RF IC: negative-market **0.0961** vs positive-market **0.0158**
- RF IC: falling rates **0.0680** vs rising rates approximately **-0.0010**
- RF IC: normal curve **0.0787** vs inverted **0.0244**
- RF IC: high VIX **0.0466** vs low VIX **0.0246**
- XGBoost and LightGBM showed related rate/curve patterns.

Caution: only 35 months; cells with <6 months are flagged; multiple-comparison inference is explicitly disclaimed.

### Milestone 3C - institutional portfolio optimizer

Objective concept:

```text
maximize alpha'w - risk_aversion * w'Σw - turnover_penalty * |w - w_prev|_1
```

Constraints:
- long only
- fully invested
- max single-name weight 10%
- explicit turnover caps
- covariance: point-in-time 252-day Ledoit-Wolf shrinkage, minimum 126 observations
- turnover measured against actual drifted pre-trade weights

Validation selected:
- Ridge: conservative
- Random Forest: conservative
- XGBoost: conservative
- LightGBM: low_turnover
- primary validation-selected optimized alpha source: Ridge

Historical optimized results:
- RF optimized: return **21.12%**, vol **13.73%**, Sharpe **1.464**, max DD **-16.57%**, turnover **4.28x**
- XGB optimized: return **18.41%**, Sharpe **1.349**, max DD **-14.57%**, turnover **5.39x**
- LGB optimized: return **20.31%**, Sharpe **1.349**, turnover **4.69x**
- Ridge optimized: return **13.89%**, Sharpe **1.086**, turnover **5.92x**

RF optimization reduced its heuristic turnover from **11.04x -> 4.28x** and raised Sharpe from **1.157 -> 1.464**, at the cost of lower raw return.

Average optimized effective N: roughly **12-13**; 10% position cap frequently binds. Solver success was 100%; the only expected turnover-cap relaxation is the first deployment from cash.

### Milestone 3D - genuine 2026 prospective holdout

Successful run:

```powershell
quant-research run-milestone3d --as-of 2026-09-01
```

Market coverage: **60/60 tickers**, latest completed bar **2026-08-31**.

Research methodology is frozen as of **1 September 2026**.

Only **7** 2026 month-end labels had matured sufficiently for Rank IC. Early prospective Rank IC is negative for every learned model:
- Ridge: **-0.0656**
- Random Forest: **-0.0945**
- XGBoost: **-0.0984**
- LightGBM: **-0.0996**

Portfolio results Jan-Aug 2026 (annualized metrics are provisional because the sample is partial-year):
- Momentum: total return **27.20%**, annualized Sharpe **1.672**
- RF heuristic: total return **27.12%**, Sharpe **1.648**
- Ridge optimized: total return **12.45%**, Sharpe **1.756**, max DD **-6.98%**
- Equal weight: total return **12.37%**, Sharpe **1.573**
- RF optimized: total return **10.34%**, Sharpe **1.298**
- XGB optimized: total return **9.72%**, Sharpe **1.249**
- LGB optimized: total return **5.61%**, Sharpe **0.675**
- Ridge heuristic: total return **0.80%**, Sharpe **0.166**

Interpretation: historical nonlinear ranking did not generalize in the first 7 matured prospective months. This is **negative early evidence, not a final statistical verdict**. Do not change the research specification in response.

---

## 10. Engineering issues encountered and permanent fixes

### A. Python version mismatch
**Failure:** project required Python >=3.12 but first `.venv` used Python 3.11.9.

**Fix:** existing Astral CPython 3.12.11 was used to recreate `.venv`. Dependencies remain installed inside the project virtual environment, not globally.

### B. Yahoo adjusted OHLC false validation failures
**Failure:** 66 historical rows failed exact OHLC inequalities because adjusted floating-point values differed by ~1e-14.

**Fix:** strict validation retained but equality uses `numpy.isclose` tolerance; genuine OHLC violations are still rejected and diagnostic samples are printed.

### C. Fama-French PeriodIndex compatibility
**Failure:** pandas rejected `pd.to_datetime(PeriodIndex)`.

**Fix:** if the reader returns `PeriodIndex`, use `.to_timestamp()`; otherwise normal datetime conversion.

### D. pandas merge dtype mismatch
**Failure:** `datetime64[s]` and `datetime64[us]` keys caused `merge_asof` failure.

**Fix:** market, macro and factor dates are normalized to `datetime64[ns]` before merges.

### E. Macro holiday missingness
**Failure:** equity sessions on some Treasury/bond holidays inherited same-day FRED NaNs, collapsing complete-feature cross-sections.

**Fix:** macro series use backward-only last-known observation. Future filling is prohibited.

### F. Non-self-financing backtester
**Failure:** early backtester implicitly held target weights constant between monthly rebalances without charging the daily trades needed to do so.

**Fix:** weights now drift naturally with asset returns; only target dates rebalance; turnover is computed versus drifted pre-trade weights.

### G. Cross-partition label leakage
**Failure:** late-December train/validation observations had 21-day labels that extended into the next partition.

**Fix:** purged splits retain rows only when label end dates remain inside the same partition. 1,260 rows removed from each split.

### H. Fake month-end from purged data
**Failure:** a purged endpoint such as 1 Dec 2025 could be mistaken for “December month-end.”

**Fix:** true month-end signal dates are identified from the full trading calendar before partition filtering.

### I. Matured-label chronology
**Risk:** a label finishing on the current signal close should not be treated as known before generating that signal.

**Fix:** training rule is strictly `label_end_date < signal_date`.

### J. Corporate action / anomaly review
- DHR +~61% adjusted move on 5 Jul 2016 aligned with the Fortive separation; flagged rather than blindly removed.
- AMD had a zero-volume record on 2 Jan 2015; handled as a data-quality/eligibility anomaly.

### K. SHAP dependency after 3A upgrade
**Failure:** test suite failed with `ModuleNotFoundError: shap` after code replacement.

**Fix:** rerun editable dependency install: `python -m pip install -e ".[dev,ml]"`. SHAP deprecation warnings from its internal matplotlib usage were non-fatal.

### L. Fed-funds regime quantile tie edge case
**Issue:** rolling low/high tercile thresholds can coincide for step-like policy rates; equality could be incorrectly pushed to `high`.

**Fix:** tied-threshold equality now maps to neutral `medium`. This was a latent edge case and did not materially change the existing 2023-25 3B result.

### M. Optimizer first-deployment turnover feasibility
**Issue:** going from cash to 100% invested necessarily has turnover >=1, so a 0.4/0.5 turnover cap may be impossible at initial deployment.

**Fix:** calculate minimum feasible turnover by linear programming; relax the cap only to that mathematical minimum and record `turnover_cap_relaxed`. No post-initial relaxations occurred in 3C.

### N. In-progress 2026 daily bar
**Failure:** running 3D on 1 Sep 2026 while the US session was still open returned incomplete Yahoo daily rows for BLK, CRM and DHR with genuine OHLC inconsistencies.

**Fix:** `--as-of` is an **exclusive market-data cutoff**; unfinished as-of-session rows are dropped before validation. The successful 1 Sep run therefore used data through 31 Aug 2026.

---

## 11. Current research conclusion

The platform produced a deliberately mixed result rather than a manufactured “ML beats the market” result:

1. Momentum remains a very strong benchmark.
2. Ridge showed unstable full-cross-sectional ranking despite high concentrated historical portfolio returns.
3. Random Forest produced the most persistent positive historical Rank IC among nonlinear models, but confidence intervals included zero.
4. All three nonlinear models selected momentum + volatility + liquidity, and SHAP showed strong momentum-liquidity / momentum-volatility interactions.
5. Regime diagnostics suggest strong temporal/state dependence, especially around rate direction, yield-curve shape and market trend, but the sample is too small for regime switching.
6. Ledoit-Wolf / turnover-aware optimization materially improved implementability and, especially for Random Forest, historical risk-adjusted efficiency.
7. In the first genuinely prospective 2026 period, all learned models had negative Rank IC. This is the strongest evidence that historical fit was not stable enough to claim persistent alpha.
8. The methodology is now frozen. Future 2026+ data should extend the prospective record, not trigger retrospective tuning.

---

## 12. What must NOT be done next

- Do not add LSTM/Transformer/CatBoost simply to search for a winner.
- Do not alter feature definitions because the 2026 IC is negative.
- Do not change RF depth/XGB parameters/LGB leaves because of 2026.
- Do not change the 10% position cap or optimizer penalties because another choice would have looked better in 2026.
- Do not reclassify 2023-2025 as an untouched test set.
- Do not publish raw Yahoo parquet data in the public repository.

---

## 13. NEXT STEP: Milestone 4A - product boundary

Build a drop-in upgrade with two components.

### A. FastAPI research API
Suggested module layout:

```text
src/quant_platform/api/
├── app.py
├── schemas.py
├── dependencies.py
├── artifacts.py
└── routers/
    ├── overview.py
    ├── models.py
    ├── performance.py
    ├── risk.py
    ├── explainability.py
    ├── regimes.py
    ├── optimization.py
    ├── holdout.py
    └── methodology.py
```

Suggested endpoints:

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

Requirements:
- Pydantic response schemas
- artifact existence/schema validation
- deterministic API serialization
- API tests
- FastAPI `/docs` / OpenAPI
- no model retraining inside HTTP requests
- no raw OHLCV exposed publicly

### B. Static web-data exporter
Add a CLI such as:

```powershell
quant-research export-web
```

It should transform research artifacts into small stable public JSON files:

```text
web/public/data/
├── overview.json
├── strategies.json
├── performance.json
├── rank_ic.json
├── models.json
├── shap.json
├── regimes.json
├── optimization.json
├── holdout.json
└── methodology.json
```

The future dashboard should load these static files instantly; FastAPI is supplemental rather than a cold-start dependency.

---

## 14. Product-side roadmap after 4A

### 4B - Next.js institutional dashboard
Status: **complete in repository version 0.10.0**. The dashboard is implemented
under `web/`, loads the ten deterministic JSON exports as its primary source,
and preserves the full negative 2026 Rank IC record. FastAPI remains
supplemental. The research methodology was not redesigned or retuned.

Preferred stack:
- Next.js + TypeScript
- Tailwind CSS
- shadcn/ui
- Apache ECharts or Plotly.js

Proposed sidebar/pages:

```text
Overview
Models
Performance
Risk
Explainability
Regimes
Optimization
2026 Holdout
Data Quality
Methodology
```

The UI should visibly show the negative 2026 Rank IC; do not cherry-pick.

### 4C - Docker
Status: **complete in repository version 0.11.0**. The root Compose file builds
independent, health-checked API and dashboard containers. The API receives the
derived artifact directory read-only, the dashboard remains static-JSON-first,
and neither container executes or retunes the research pipeline.

Local target:

```text
localhost:3000  dashboard
localhost:8000  FastAPI
localhost:8000/docs
```

`docker compose up` should start both frontend and API.

### 4D - deployment / CI-CD

Status: **implementation complete in repository version 0.12.0; external account connection pending user action.**

Implemented:
- GitHub Actions tests Python 3.12/3.13, Ruff, pytest, deterministic web exports, Next.js lint/build, and production container builds.
- `render.yaml` defines the supplemental API as a Docker web service with `/health`, CI-gated auto-deploy, and an explicit artifact root.
- `web/vercel.json` identifies the dashboard as Next.js; Vercel must be configured with repository root directory `web`.
- The API Docker command honors Render's `PORT` and keeps local Compose on port 8000.
- The normal wheel-style install is retained. `QUANT_ARTIFACT_ROOT=/app/artifacts` permanently replaces the fragile editable-install path workaround.
- A manual `deployment-smoke` workflow verifies the dashboard, API health, and overview schema after publishing.
- Raw provider snapshots, local credentials, build output, and provider-local state are ignored.
- Next.js was updated narrowly from 16.2.6 to 16.3.4 after the 4D production dependency audit identified newly disclosed vulnerabilities; the post-update production audit reports zero vulnerabilities.

Provider limits were rechecked against official documentation on 1 Sep 2026:

**Frontend - Vercel Hobby**
- $0/month
- automatic CI/CD, CDN
- Hobby is for personal/non-commercial use
- https://vercel.com/pricing
- https://vercel.com/legal/terms

**FastAPI - Render Free**
- 750 free instance hours/workspace/month
- free services spin down after 15 minutes idle and take about a minute to wake
- use only as supplemental API; dashboard should not block on it
- https://render.com/docs/free

**CI - GitHub Actions**
- standard GitHub-hosted runners are free/unlimited for public repositories
- https://docs.github.com/en/actions/reference/runners/github-hosted-runners

**Optional experiment registry - Neon Free Postgres**
- 10 projects; 50 CU-hours/month/project; 0.5 GB/project; 5 GB egress/month at the researched plan
- https://neon.com/blog/new-usage-based-pricing

**Static alternative - Cloudflare Pages**
- Free plan currently documents 500 builds/month
- https://developers.cloudflare.com/pages/platform/limits/

**All-Python fallback - Streamlit Community Cloud**
- free and connects directly to GitHub; easier but less polished than Next.js for this project
- https://docs.streamlit.io/deploy/streamlit-community-cloud

Free-plan limits can change. Recheck provider pages again before a later migration or paid-plan decision.

### 4E - experiment registry
Next store experiment id, git commit, dataset/config hashes, model, parameters and metrics. Do not make this a prerequisite for the dashboard or alter any frozen research choice.

### 4F - frozen monthly refresh
Automate incremental prospective updates without altering methodology. Dashboard should display methodology freeze date, latest data date and hashes.

---

## 15. Public GitHub / data rules

The public repository may contain:
- acquisition code
- feature/model/backtest code
- tests
- configs
- methodology
- derived summary metrics/charts
- dashboard JSON

Do **not** commit raw Yahoo market data. The yfinance project states that it is intended for research/education and that Yahoo Finance data are intended for personal use; users should download their own source data.

Recommended `.gitignore` protection:

```text
data/raw/*.parquet
data/holdout2026/raw/*.parquet
```

---

## 16. Files to inspect first in the new chat

After the user uploads the whole code folder, inspect:

```text
pyproject.toml
configs/research.yaml
src/quant_platform/cli.py
src/quant_platform/ml/holdout.py
src/quant_platform/ml/optimizer.py
artifacts/milestone3d/summary.json
artifacts/milestone3d/frozen_specifications.json
artifacts/milestone3d/portfolio/performance_metrics.csv
```

Then build 4A without touching the frozen research choices.

---

## 17. Final handoff instruction

**Treat this repository as a completed research engine with a deployed-product configuration.** Do not restart the quantitative research from scratch. Preserve the findings, including disappointing holdout results. The next work should add experiment metadata and reproducibility around the already-frozen research engine.
