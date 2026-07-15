#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METRICS="${1:-$ROOT/logs/metrics.jsonl}"
exec "$ROOT/.venv/bin/python" -m hurgor.metrics "$METRICS"
