#!/usr/bin/env bash
# Generate temporal PubMedBERT CLS embeddings from protein descriptions.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$PYTHON_BIN" "$HERE/run_pfp_temporal_text.py" --help
fi

exec "$PYTHON_BIN" "$HERE/run_pfp_temporal_text.py" "$@"
