#!/usr/bin/env bash
# Generate ESM-IF1 embeddings from prepared protein names and PDB structures.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GENERATOR="${HERE}/generate_structure_embeddings.py"
PYTHON_BIN="python3"
IF1_PYTHONPATH=""
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --pythonpath)
      IF1_PYTHONPATH="$2"
      shift 2
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$IF1_PYTHONPATH" ]]; then
  PYTHONPATH="$IF1_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
    exec "$PYTHON_BIN" "$GENERATOR" "${FORWARD[@]}"
else
  exec "$PYTHON_BIN" "$GENERATOR" "${FORWARD[@]}"
fi
