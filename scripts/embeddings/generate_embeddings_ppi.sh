#!/usr/bin/env bash
# Generate STRING PPI embeddings for the prepared proteins.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GENERATOR="${HERE}/generate_ppi_embeddings.py"
PYTHON_BIN="python3"
PPI_PYTHONPATH=""
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --pythonpath)
      PPI_PYTHONPATH="$2"
      shift 2
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$PPI_PYTHONPATH" ]]; then
  PYTHONPATH="$PPI_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
    exec "$PYTHON_BIN" "$GENERATOR" "${FORWARD[@]}"
else
  exec "$PYTHON_BIN" "$GENERATOR" "${FORWARD[@]}"
fi
