#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMES="${1:-2250}"
shift || true
exec "$ROOT/.venv/bin/python" -m hurgor.client --base-url "${HURGOR_ENDURANCE_URL:-http://127.0.0.1:8765}" --max-frames "$FRAMES" "$@"
