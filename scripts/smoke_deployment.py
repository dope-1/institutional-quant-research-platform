from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


def _request_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _request_ok(url: str, *, timeout: float) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")


def _retry(action, *, attempts: int, delay: float) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(delay)
    raise RuntimeError(f"Deployment smoke check failed after {attempts} attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test deployed dashboard and API URLs.")
    parser.add_argument("--dashboard-url", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--delay", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    dashboard_url = args.dashboard_url.rstrip("/") + "/"
    api_url = args.api_url.rstrip("/")

    _retry(
        lambda: _request_ok(dashboard_url, timeout=args.timeout),
        attempts=args.attempts,
        delay=args.delay,
    )
    health = _retry(
        lambda: _request_json(f"{api_url}/health", timeout=args.timeout),
        attempts=args.attempts,
        delay=args.delay,
    )
    if health.get("status") != "ok":
        raise RuntimeError(f"API health payload is not healthy: {health}")
    overview = _retry(
        lambda: _request_json(f"{api_url}/api/v1/overview", timeout=args.timeout),
        attempts=args.attempts,
        delay=args.delay,
    )
    if overview.get("schema_version") != "1.0.0":
        raise RuntimeError(f"Unexpected overview schema version: {overview.get('schema_version')}")

    print("Deployment smoke test passed: dashboard, API health, and overview are reachable.")


if __name__ == "__main__":
    main()
