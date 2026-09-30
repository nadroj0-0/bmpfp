#!/usr/bin/env bash
# Train the upstream PFP model from prepared data and embedding paths.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
Usage:
  run_pfp_benchmark.sh \
    --prepared-data-dir DIR --sequence-embedding-dir DIR \
    --pfp-root DIR --obo-file FILE --output-dir DIR \
    [--text-embedding-dir DIR] [--structure-embedding-dir DIR] \
    [--ppi-embedding-dir DIR] [--aspect BPO|CCO|MFO]... \
    [--mode full|sequence-only|sequence-text|sequence-structure|sequence-ppi] \
    [--seed N] [--num-workers N] [--ia-file-dir DIR]

The wrapper presents the supplied embedding directories in the upstream PFP
layout and runs train.py with the configured model settings.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
require_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"; }

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREPARED_DATA_DIR=""
SEQUENCE_EMBEDDING_DIR=""
TEXT_EMBEDDING_DIR=""
STRUCTURE_EMBEDDING_DIR=""
PPI_EMBEDDING_DIR=""
PFP_ROOT=""
OBO_FILE=""
OUTPUT_DIR=""
MODE="full"
SEED=42
NUM_WORKERS=0
IA_FILE_DIR=""
ASPECTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepared-data-dir) require_value "$@"; PREPARED_DATA_DIR="$2"; shift 2 ;;
    --sequence-embedding-dir) require_value "$@"; SEQUENCE_EMBEDDING_DIR="$2"; shift 2 ;;
    --text-embedding-dir) require_value "$@"; TEXT_EMBEDDING_DIR="$2"; shift 2 ;;
    --structure-embedding-dir) require_value "$@"; STRUCTURE_EMBEDDING_DIR="$2"; shift 2 ;;
    --ppi-embedding-dir) require_value "$@"; PPI_EMBEDDING_DIR="$2"; shift 2 ;;
    --pfp-root) require_value "$@"; PFP_ROOT="$2"; shift 2 ;;
    --obo-file) require_value "$@"; OBO_FILE="$2"; shift 2 ;;
    --output-dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --aspect) require_value "$@"; ASPECTS+=("$2"); shift 2 ;;
    --mode) require_value "$@"; MODE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --num-workers) require_value "$@"; NUM_WORKERS="$2"; shift 2 ;;
    --ia-file-dir) require_value "$@"; IA_FILE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$PREPARED_DATA_DIR" ]] || die "--prepared-data-dir is required"
[[ -n "$SEQUENCE_EMBEDDING_DIR" ]] || die "--sequence-embedding-dir is required"
[[ -n "$PFP_ROOT" ]] || die "--pfp-root is required"
[[ -n "$OBO_FILE" ]] || die "--obo-file is required"
[[ -n "$OUTPUT_DIR" ]] || die "--output-dir is required"
[[ "$MODE" =~ ^(full|sequence-only|sequence-text|sequence-structure|sequence-ppi)$ ]] || \
  die "invalid --mode: $MODE"

case "$MODE" in
  full)
    [[ -n "$TEXT_EMBEDDING_DIR" ]] || die "--text-embedding-dir is required in full mode"
    [[ -n "$STRUCTURE_EMBEDDING_DIR" ]] || die "--structure-embedding-dir is required in full mode"
    [[ -n "$PPI_EMBEDDING_DIR" ]] || die "--ppi-embedding-dir is required in full mode"
    ;;
  sequence-text) [[ -n "$TEXT_EMBEDDING_DIR" ]] || die "--text-embedding-dir is required" ;;
  sequence-structure) [[ -n "$STRUCTURE_EMBEDDING_DIR" ]] || die "--structure-embedding-dir is required" ;;
  sequence-ppi) [[ -n "$PPI_EMBEDDING_DIR" ]] || die "--ppi-embedding-dir is required" ;;
esac

if [[ ${#ASPECTS[@]} -eq 0 ]]; then ASPECTS=(BPO CCO MFO); fi
for aspect in "${ASPECTS[@]}"; do
  [[ "$aspect" =~ ^(BPO|CCO|MFO)$ ]] || die "invalid aspect: $aspect"
done

mkdir -p "$OUTPUT_DIR"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pfp-training.XXXXXX")"
trap 'rm -rf -- "$WORK_DIR"' EXIT
STAGED_DATA="$WORK_DIR/data"
EMPTY_MODALITY="$WORK_DIR/empty_modality"
mkdir -p "$STAGED_DATA/embedding_cache" "$EMPTY_MODALITY"

PREPARED_DATA_DIR="$(cd "$PREPARED_DATA_DIR" && pwd -P)"
while IFS= read -r -d '' source_path; do
  name="${source_path##*/}"
  [[ "$name" == "embedding_cache" ]] && continue
  ln -s "$source_path" "$STAGED_DATA/$name"
done < <(find "$PREPARED_DATA_DIR" -mindepth 1 -maxdepth 1 -print0)

ACTIVE_TEXT="$EMPTY_MODALITY"
ACTIVE_STRUCTURE="$EMPTY_MODALITY"
ACTIVE_PPI="$EMPTY_MODALITY"
case "$MODE" in
  full)
    ACTIVE_TEXT="$TEXT_EMBEDDING_DIR"
    ACTIVE_STRUCTURE="$STRUCTURE_EMBEDDING_DIR"
    ACTIVE_PPI="$PPI_EMBEDDING_DIR"
    ;;
  sequence-text) ACTIVE_TEXT="$TEXT_EMBEDDING_DIR" ;;
  sequence-structure) ACTIVE_STRUCTURE="$STRUCTURE_EMBEDDING_DIR" ;;
  sequence-ppi) ACTIVE_PPI="$PPI_EMBEDDING_DIR" ;;
esac

SEQUENCE_EMBEDDING_DIR="$(cd "$SEQUENCE_EMBEDDING_DIR" && pwd -P)"
if [[ "$ACTIVE_TEXT" != "$EMPTY_MODALITY" ]]; then
  ACTIVE_TEXT="$(cd "$ACTIVE_TEXT" && pwd -P)"
fi
if [[ "$ACTIVE_STRUCTURE" != "$EMPTY_MODALITY" ]]; then
  ACTIVE_STRUCTURE="$(cd "$ACTIVE_STRUCTURE" && pwd -P)"
fi
if [[ "$ACTIVE_PPI" != "$EMPTY_MODALITY" ]]; then
  ACTIVE_PPI="$(cd "$ACTIVE_PPI" && pwd -P)"
fi

ln -s "$SEQUENCE_EMBEDDING_DIR" "$STAGED_DATA/embedding_cache/prott5"
ln -s "$ACTIVE_TEXT" "$STAGED_DATA/embedding_cache/exp_text_embeddings"
ln -s "$ACTIVE_STRUCTURE" "$STAGED_DATA/embedding_cache/IF1"
ln -s "$ACTIVE_PPI" "$STAGED_DATA/embedding_cache/ppi"

for aspect in "${ASPECTS[@]}"; do
  command=(
    "$PYTHON_BIN" "$PFP_ROOT/train.py"
    --single
    --seq-model prott5
    --fusion-types gated_bilinear
    --aspects "$aspect"
    --data-dir "$STAGED_DATA"
    --obo-file "$OBO_FILE"
    --output-base "$OUTPUT_DIR"
    --use-late-fusion
    --late-output-mode hybrid
    --aux-loss-weight 0.8
    --modality-dropout 0.1
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
  )
  if [[ -n "$IA_FILE_DIR" ]]; then command+=(--ia-file-dir "$IA_FILE_DIR"); fi
  "${command[@]}"
done
