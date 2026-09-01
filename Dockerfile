# syntax=docker/dockerfile:1

FROM python:3.14-slim AS api

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QUANT_ARTIFACT_ROOT=/app/artifacts

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir . \
    && addgroup --system quant \
    && adduser --system --ingroup quant quant

# The API validates and exposes only whitelisted, derived research artifacts.
# Compose overlays the local artifacts directory read-only for easy refreshes.
COPY --chown=quant:quant artifacts ./artifacts

USER quant

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import os, urllib.request; port = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4)" || exit 1

CMD ["sh", "-c", "exec quant-research serve-api --host 0.0.0.0 --port ${PORT:-8000}"]
