# Milestone 4B Dashboard

This Next.js dashboard is a read-only presentation layer over the frozen
Milestone 4A JSON exports in `public/data/`. It does not train models, modify
research settings, or require FastAPI at page load.

## Run on Windows PowerShell

From the repository root:

```powershell
quant-research export-web
cd web
npm install
npm run dev
```

Open `http://localhost:3000`.

The FastAPI service at `http://127.0.0.1:8000` is optional for the dashboard. To
inspect the API separately, run `quant-research serve-api` in another terminal
and open `http://127.0.0.1:8000/docs`.

## Verify a production build

```powershell
npm run lint
npm run build
npm start
```

`npm start` serves the production build at `http://localhost:3000`.

## Dashboard routes

- `/` — overview and visible prospective evidence
- `/models` — frozen model registry
- `/performance` — historical, optimized, and holdout performance
- `/risk` — partial-year risk surface
- `/explainability` — post-hoc SHAP summaries
- `/regimes` — conditional historical diagnostics
- `/optimization` — frozen constraints and concentration
- `/holdout` — complete 2026 prospective record
- `/data-quality` — audit and review flags
- `/methodology` — frozen methodology and chronology safeguards

After the research exporter rewrites `public/data/`, refresh the dashboard. No
frontend code generation step is required.
