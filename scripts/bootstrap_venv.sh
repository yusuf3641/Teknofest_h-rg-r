#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.12 bulunamadı. PYTHON_BIN ile tam yolu verin." >&2
  exit 2
fi
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  VERSION="$($ROOT/.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$VERSION" != "3.12" ]]; then
    echo "Mevcut .venv Python $VERSION kullanıyor; otomatik silinmedi. Yedekleyip yeniden deneyin." >&2
    exit 3
  fi
else
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements-runtime.lock"
"$ROOT/.venv/bin/python" -m pip install --no-deps -e "$ROOT"
