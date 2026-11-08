#!/usr/bin/env bash
# Run the full paired experiment with the final settings and keep a log.
#
#   bash scripts/run_experiment.sh
#   OUTPUT_DIR=data/processed/exp_llm LLM_RATE=0.3 LLM_BACKEND=groq bash scripts/run_experiment.sh
#
# Re-running the same command after an interruption resumes: seeds whose
# results already exist in OUTPUT_DIR are reused, not recomputed.
set -euo pipefail

cd "$(dirname "$0")/.."
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/experiment}"
SEEDS="${SEEDS:-17 29 41 53 67}"
LLM_RATE="${LLM_RATE:-0}"
LLM_BACKEND="${LLM_BACKEND:-none}"

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1
echo "Logging to $LOG"

# shellcheck disable=SC2086
python -m python.search.run_comparison \
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
    --fdr-q 0.05 \
    --ic-test-method hansen_hodrick \
    --llm-rate "$LLM_RATE" \
    --llm-backend "$LLM_BACKEND" 2>&1 | tee "$LOG"

python -m python.search.report_comparison \
    --input "$OUTPUT_DIR/comparison_summary.json" 2>&1 | tee -a "$LOG"
