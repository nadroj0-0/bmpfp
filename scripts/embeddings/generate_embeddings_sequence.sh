#!/usr/bin/env bash
# Generate ProtT5 embeddings from a protein FASTA file.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  generate_embeddings_sequence.sh \
    --extractor /path/to/PFP/scripts/extract_prott5_embeddings.py \
    --fasta-file /path/to/proteins.fasta \
    --output-dir /path/to/embeddings/prott5 \
    --model-dir /path/to/local/prot_t5_xl_half_uniref50-enc \
    [--python-bin python3] [--batch-size 8] [--max-length 1024] \
    [--device cpu|cuda|cuda:N]

Runs the supplied ProtT5 extractor for the requested protein FASTA file.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
require_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"; }

EXTRACTOR=""
FASTA_FILE=""
OUTPUT_DIR=""
MODEL_DIR=""
PYTHON_BIN="python3"
BATCH_SIZE="8"
MAX_LENGTH="1024"
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extractor) require_value "$@"; EXTRACTOR="$2"; shift 2 ;;
    --fasta-file) require_value "$@"; FASTA_FILE="$2"; shift 2 ;;
    --output-dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --model-dir) require_value "$@"; MODEL_DIR="$2"; shift 2 ;;
    --python-bin) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --max-length) require_value "$@"; MAX_LENGTH="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$EXTRACTOR" ]] || die "--extractor is required"
[[ -n "$FASTA_FILE" ]] || die "--fasta-file is required"
[[ -n "$OUTPUT_DIR" ]] || die "--output-dir is required"
[[ -n "$MODEL_DIR" ]] || die "--model-dir is required"
[[ "$DEVICE" == "cpu" || "$DEVICE" == "cuda" || "$DEVICE" =~ ^cuda:[0-9]+$ ]] || \
  die "--device must be cpu, cuda, or cuda:N"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON_BIN" "$EXTRACTOR" \
  --fasta_file "$FASTA_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --device "$DEVICE"
