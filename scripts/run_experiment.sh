#!/usr/bin/env bash
# Run the full paired experiment with the final settings and keep a log.
#
#   C4, evolution vs random (no LLM, no API key):
#     bash scripts/run_experiment.sh
#
#   E2, adds LLM-guided evolution as a third method. Run it into the SAME
#   folder: the evolution and random results of C4 are reused, only the LLM
#   method runs.
#     LLM_RATE=0.3 LLM_BACKEND=ollama bash scripts/run_experiment.sh
#     LLM_RATE=0.3 LLM_BACKEND=openrouter LLM_MODEL=<model id> bash scripts/run_experiment.sh
#
#   E3 (optional), also runs the LLM without retrieval on two seeds:
#     LLM_RATE=0.3 LLM_BACKEND=ollama ABLATION_SEEDS="17 29" bash scripts/run_experiment.sh
#
# Each run ends by writing comparison_summary.html and tables.md (Tables 2 and 3).
#
#   ENGINE=cpp evaluates fitness with the compiled C++ engine (build it first,
#   see README); results match the Python engine to rounding, and runs made
#   with either engine are reused.
#
# Re-running the same command after an interruption resumes: runs whose
# results already exist in OUTPUT_DIR are reused, not recomputed.
set -euo pipefail

cd "$(dirname "$0")/.."
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/experiment}"
SEEDS="${SEEDS:-17 29 41 53 67}"
LLM_RATE="${LLM_RATE:-0}"
LLM_BACKEND="${LLM_BACKEND:-none}"
LLM_MODEL="${LLM_MODEL:-}"
MODEL_ARGS=()
if [ -n "$LLM_MODEL" ]; then MODEL_ARGS=(--llm-model "$LLM_MODEL"); fi
ABLATION_SEEDS="${ABLATION_SEEDS:-}"
ENGINE="${ENGINE:-python}"
ABLATION_ARGS=()
# shellcheck disable=SC2206
if [ -n "$ABLATION_SEEDS" ]; then ABLATION_ARGS=(--ablation-seeds $ABLATION_SEEDS); fi

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1
echo "Logging to $LOG"

# shellcheck disable=SC2086
python -m python.run_pipeline \
    --output-dir "$OUTPUT_DIR" \
    --seeds $SEEDS \
    --population-size 32 \
    --generations 10 \
    --elite-count 4 \
    --mutation-rate 0.7 \
    --crossover-rate 0.3 \
    --horizon 20 \
    --max-depth 8 \
    --top-k 10 \
    --min-coverage 0.5 \
    --min-ic-day-share 0.5 \
    --fdr-q 0.05 \
    --ic-test-method hansen_hodrick \
    --llm-rate "$LLM_RATE" \
    --llm-backend "$LLM_BACKEND" \
    --engine "$ENGINE" \
    "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}" \
    "${ABLATION_ARGS[@]+"${ABLATION_ARGS[@]}"}" 2>&1 | tee "$LOG"

