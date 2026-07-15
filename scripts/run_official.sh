#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" != "--confirm-official" ]]; then
  echo "Refusing to POST. Use --confirm-official after preflight and team approval." >&2
  exit 2
fi
shift
"$ROOT/.venv/bin/python" -m hurgor.preflight
exec "$ROOT/.venv/bin/python" -m hurgor.client "$@"
