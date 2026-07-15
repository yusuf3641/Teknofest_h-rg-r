#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="${1:-$ROOT/wheelhouse}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
mkdir -p "$WHEELHOUSE"
"$PYTHON_BIN" -m pip wheel --wheel-dir "$WHEELHOUSE" -r "$ROOT/requirements-runtime.lock"
exec "$PYTHON_BIN" -m pip wheel --no-deps --wheel-dir "$WHEELHOUSE" "$ROOT"
