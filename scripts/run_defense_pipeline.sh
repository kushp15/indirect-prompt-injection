#!/usr/bin/env bash
# =============================================================================
# run_defense_pipeline.sh — PromptShield Multi-Layer Defense Evaluation
# =============================================================================
# Edit the CONFIG section below, then run:
#   ./scripts/run_defense_pipeline.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ===========================================================================
# CONFIG — edit these values before running
# ===========================================================================

MODEL_NAME="llama3:instruct"   # Ollama model tag for the main agent
SETTING="base"                # Attack difficulty: "base" or "enhanced"
PROMPT_TYPE="InjecAgent"      # Prompt template:  "InjecAgent" or "hwchase17_react"

# Defense layers — set to true/false to toggle each one
ENABLE_LAYER1=true            # Layer 1: structural delimiters (zero extra LLM calls)
ENABLE_LAYER2=true           # Layer 2: router LLM firewall   (1 extra call/case)
ENABLE_LAYER3=true           # Layer 3: LLM-as-a-judge        (1 extra call/case)

# Optional: use a lighter/faster model for the defense layers
# Leave blank to use the same MODEL_NAME for all three layers
FIREWALL_MODEL="llama3:instruct"             # e.g. "gemma2:2b"   — used by Layer 2
JUDGE_MODEL="llama3:instruct"                # e.g. "gemma2:2b"   — used by Layer 3

# How many test cases to run per attack type (dh + ds).
# Leave blank ("") or set to 0 to run ALL cases.
# Set to a number (e.g. 20) to do a quick smoke test first.
MAX_CASES="5"

# ===========================================================================

# ---------------------------------------------------------------------------
# Optional: source .env from repo root
# ---------------------------------------------------------------------------
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -o allexport
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.env"
    set +o allexport
fi

# ---------------------------------------------------------------------------
# Sanity check: Ollama must be reachable
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach Ollama at ${OLLAMA_BASE_URL}."
    echo "       Start it with:  ollama serve"
    exit 1
fi
echo "Ollama is running at ${OLLAMA_BASE_URL}."
echo ""

# ---------------------------------------------------------------------------
# Build argument list from CONFIG
# ---------------------------------------------------------------------------
ARGS=(
    --model_name  "${MODEL_NAME}"
    --setting     "${SETTING}"
    --prompt_type "${PROMPT_TYPE}"
)

[[ "${ENABLE_LAYER1}" == true ]] && ARGS+=(--enable_layer1)
[[ "${ENABLE_LAYER2}" == true ]] && ARGS+=(--enable_layer2)
[[ "${ENABLE_LAYER3}" == true ]] && ARGS+=(--enable_layer3)

[[ -n "${FIREWALL_MODEL}" ]] && ARGS+=(--firewall_model "${FIREWALL_MODEL}")
[[ -n "${JUDGE_MODEL}"    ]] && ARGS+=(--judge_model    "${JUDGE_MODEL}")

if [[ -n "${MAX_CASES}" && "${MAX_CASES}" -gt 0 ]]; then
    ARGS+=(--max_cases "${MAX_CASES}")
fi

# ---------------------------------------------------------------------------
# Print summary and run
# ---------------------------------------------------------------------------
echo "=== PromptShield Defense Pipeline ==="
echo "  Model      : ${MODEL_NAME}"
echo "  Setting    : ${SETTING}"
echo "  Prompt type: ${PROMPT_TYPE}"
echo "  Layers     : L1=${ENABLE_LAYER1}  L2=${ENABLE_LAYER2}  L3=${ENABLE_LAYER3}"
[[ -n "${FIREWALL_MODEL}" ]] && echo "  Firewall   : ${FIREWALL_MODEL}"
[[ -n "${JUDGE_MODEL}"    ]] && echo "  Judge      : ${JUDGE_MODEL}"
echo "  Max cases  : ${MAX_CASES:-all}"
echo "  Cache      : disabled (fresh run every time)"
echo ""

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

python3 src/evaluate_defense_pipeline.py "${ARGS[@]}"
