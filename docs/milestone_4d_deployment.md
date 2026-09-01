# Milestone 4D — Deployment and CI/CD

Milestone 4D makes the frozen research product ready for a public Git repository,
Vercel dashboard deployment, and supplemental Render API deployment. It changes
no feature, label, model, portfolio, optimizer, regime, or holdout rule.

## Architecture

| Boundary | Service | Source | Role |
| --- | --- | --- | --- |
| Primary UI | Vercel | `web/` | Serves the Next.js dashboard and committed public JSON |
| Supplemental API | Render | root `Dockerfile` | Serves validated derived artifacts and OpenAPI |
| CI | GitHub Actions | `.github/workflows/ci.yml` | Tests code, exports, dashboard, and images |
| Production check | GitHub Actions | `.github/workflows/deployment-smoke.yml` | Checks the two deployed URLs on demand |

The dashboard does not call the API to render its pages. A Render cold start
therefore cannot block the primary UI.

## 1. Prepare and publish the GitHub repository

From the repository root, confirm that ignored files are not staged:

```powershell
git status --short
git check-ignore data\raw\market_ohlcv.parquet
git check-ignore data\holdout2026\raw\market_ohlcv.parquet
git check-ignore .env
```

If this directory is not already a Git repository:

```powershell
git init
git add .
git commit -m "Complete Milestone 4D deployment and CI/CD"
git branch -M main
git remote add origin https://github.com/YOUR-USER/YOUR-REPOSITORY.git
git push -u origin main
```

Create the empty GitHub repository in your own account before adding the remote.
The recommended no-cost route is public because standard GitHub-hosted Actions
runners are free for public repositories. Review every staged file before the
first push; raw source-provider snapshots and secrets must never be published.

The `ci` workflow starts automatically. It must pass before continuing.

## 2. Deploy the dashboard to Vercel

1. Sign in to Vercel with the GitHub account that owns the commits.
2. Select **Add New → Project** and import the GitHub repository.
3. Set **Root Directory** to `web`.
4. Confirm the framework preset is **Next.js**.
5. Leave environment variables empty; the dashboard uses committed
   `web/public/data/*.json`.
6. Deploy and save the resulting `https://...vercel.app` URL.

Vercel's Hobby plan is limited to personal, non-commercial use. Use an
appropriate paid plan or another host if the platform becomes commercial.

## 3. Deploy the supplemental API to Render

1. Sign in to Render and choose **New → Blueprint**.
2. Connect the same GitHub repository.
3. Render detects the root `render.yaml`; review the proposed service.
4. Confirm plan **Free**, runtime **Docker**, and health path `/health`.
5. Apply the Blueprint and wait for the first deploy.
6. Open `https://YOUR-SERVICE.onrender.com/health` and then `/docs`.

`render.yaml` waits for GitHub checks before auto-deploying. The Docker process
binds to `0.0.0.0` using Render's `PORT`. The API reads only derived artifacts
from `/app/artifacts` and does not need persistent storage.

Render Free grants 750 instance hours per workspace per month and spins a web
service down after 15 idle minutes. A later API request can take about one
minute while the service wakes. This is acceptable because the dashboard is
static-JSON-first.

## 4. Run the production smoke test

In GitHub:

1. Open **Actions → deployment-smoke → Run workflow**.
2. Enter the Vercel dashboard base URL.
3. Enter the Render API base URL.
4. Run the workflow.

The script retries long enough to tolerate a typical Render Free cold start and
then validates:

- dashboard root returns HTTP 200;
- API `/health` returns `status: ok`;
- API `/api/v1/overview` returns schema version `1.0.0`.

Equivalent local command:

```powershell
python scripts/smoke_deployment.py `
  --dashboard-url https://YOUR-DASHBOARD.vercel.app `
  --api-url https://YOUR-API.onrender.com
```

## 5. Normal update flow

1. Make a reviewed code or derived-artifact change.
2. If frozen prospective artifacts legitimately changed, run
   `quant-research export-web` and commit the deterministic JSON.
3. Push a branch and open a pull request.
4. Merge only after CI passes.
5. Vercel deploys the dashboard from `main`; Render deploys after checks pass.
6. Run `deployment-smoke` against the production URLs.

Never run a milestone research command inside a deployment. Deployment serves
only the frozen code and already-derived public artifacts.
