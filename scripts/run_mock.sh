#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv/bin/python" -m hurgor.mock_server --host 127.0.0.1 --port "${HURGOR_MOCK_PORT:-8765}" "$@"
