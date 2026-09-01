# Milestone 4C — Docker Compose

Milestone 4C packages the read-only FastAPI service and the Next.js dashboard
as separate production containers. It does not execute the research pipeline or
alter the methodology frozen after Milestone 3D.

## Services

| Service | Container process | Host URL |
| --- | --- | --- |
| `web` | Next.js standalone server | `http://localhost:3000` |
| `api` | Uvicorn through `quant-research serve-api` | `http://127.0.0.1:8000` |

Both services have independent health checks. The web container does not wait
for or fetch from the API because the Milestone 4A JSON export remains its
primary data source.

The API image contains only application source and derived research artifacts.
Compose overlays the repository's `artifacts/` directory read-only so refreshed
derived artifacts can be served without granting container write access.
`QUANT_ARTIFACT_ROOT=/app/artifacts` selects that directory explicitly. The
container therefore uses a normal package install and does not depend on the
source-tree path side effect of an editable install.

## Run

Stop any manually running development servers on ports 3000 and 8000, start
Docker Desktop, and run from the repository root:

```powershell
docker compose up --build
```

To run in the background:

```powershell
docker compose up --build -d
docker compose ps
```

Stop and remove the containers with:

```powershell
docker compose down
```

## Refresh behavior

After prospective research artifacts are legitimately extended under the frozen
rules, regenerate the public JSON and rebuild the web image:

```powershell
quant-research export-web
docker compose up --build -d
```

The rebuild is intentional: dashboard pages are statically prerendered from the
versioned JSON snapshot.

## Container hardening

- Python and Node processes run as non-root users.
- Heavy raw data, Parquet files, plots, caches, and local environments are
  excluded from build contexts.
- The API artifact bind mount is read-only.
- Images use deterministic lock-file installation and production commands.
- API and dashboard health checks run inside their respective containers.
