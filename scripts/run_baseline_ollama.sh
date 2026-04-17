#!/usr/bin/env bash
# =============================================================================
# run_baseline_ollama.sh
# =============================================================================
# Runs the standalone InjecAgent baseline evaluation with a local Ollama model.
#
# Usage:
#   ./scripts/run_baseline_ollama.sh [MODEL_NAME] [SETTING] [PROMPT_TYPE]
#
# Arguments (all optional — defaults shown below):
#   MODEL_NAME   Ollama model tag            (default: mistral:latest)
#   SETTING      'base' or 'enhanced'        (default: base)
#   PROMPT_TYPE  'InjecAgent' or             (default: InjecAgent)
#                'hwchase17_react'
#
# Examples:
#   ./scripts/run_baseline_ollama.sh
#   ./scripts/run_baseline_ollama.sh llama3:8b base InjecAgent
#   ./scripts/run_baseline_ollama.sh mistral:latest enhanced hwchase17_react
#
# Prerequisites:
#   1. Ollama running:        ollama serve
#   2. Model pulled:          ollama pull mistral:latest
#   3. Python packages:       pip install openai tqdm python-dotenv
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (parent of this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ===========================================================================
# CONFIG — edit these values before running
# ===========================================================================

MODEL_NAME="mistral:latest"   # Ollama model tag to use as the agent
SETTING="enhanced"                # Attack difficulty: "base" or "enhanced"
PROMPT_TYPE="InjecAgent"      # Prompt template: "InjecAgent" or "hwchase17_react"

# How many test cases to run per attack type (dh + ds).
# Leave blank ("") or set to 0 to run ALL 1,054 cases.
# Set to a number (e.g. 20) to do a quick smoke test first.
MAX_CASES="20"

# ===========================================================================

# ---------------------------------------------------------------------------
# Optional: load .env from the repo root
# ---------------------------------------------------------------------------
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -o allexport
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.env"
    set +o allexport
fi

# ---------------------------------------------------------------------------
# Sanity check: is Ollama reachable?
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach Ollama at ${OLLAMA_BASE_URL}."
    echo "       Start it with:  ollama serve"
    exit 1
fi
echo "Ollama is running at ${OLLAMA_BASE_URL}."

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo ""
echo "=== InjecAgent Baseline — Ollama ==="
echo "  Model      : ${MODEL_NAME}"
echo "  Setting    : ${SETTING}"
echo "  Prompt type: ${PROMPT_TYPE}"
echo "  Max cases  : ${MAX_CASES:-all}"
echo ""

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Build the max_cases flag only when a number is provided
MAX_CASES_FLAG=""
if [[ -n "${MAX_CASES}" && "${MAX_CASES}" -gt 0 ]]; then
    MAX_CASES_FLAG="--max_cases ${MAX_CASES}"
fi

python3 src/evaluate_baseline_ollama.py \
    --model_name  "${MODEL_NAME}" \
    --setting     "${SETTING}" \
    --prompt_type "${PROMPT_TYPE}" \
    --use_cache \
    ${MAX_CASES_FLAG}

echo ""
SAFE_MODEL="${MODEL_NAME//:/_}"
echo "Done. Results saved to: results/baseline_Ollama_${SAFE_MODEL}_${PROMPT_TYPE}_${SETTING}/"
