"""
evaluate_defense_pipeline.py
=============================
Full three-layer defense pipeline evaluation against the InjecAgent benchmark.

Architecture
------------
  External data
       │
       ▼
  ┌─────────────────────────────────┐
  │  Layer 1: Structural Delimiters │  layer1_delimiter.apply_layer1()
  └─────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────┐
  │  Layer 2: Router LLM Firewall   │  Layer2Firewall.screen()
  └──────────────────────────────────┘
       │ SAFE ──────────────────────► Main agent reasoning
       │ UNSAFE ────────────────────► Block + record as defended
                                           │
                                           ▼
                                  ┌─────────────────────────┐
                                  │  Layer 3: LLM-as-a-Judge│  Layer3Judge.screen_agent_output()
                                  └─────────────────────────┘
                                       │ ALIGNED ──► Execute tool safely
                                       │ MISALIGNED ► Block + record as defended

Each active layer is configurable via CLI flags.
Every output record contains raw outputs from each active layer for full
auditability and alignment with InjecAgent paper evaluation.

Usage
-----
    scripts/run_defense_pipeline.sh

Or manually:
    python3 src/evaluate_defense_pipeline.py \\
        --model_name  mistral:latest \\
        --setting     base \\
        --prompt_type InjecAgent \\
        --enable_layer1 --enable_layer2 --enable_layer3

Notes
-----
- Data is read from InjecAgent/data/ (JSON files only — no InjecAgent code).
- Results land in results/defense_<layers>_<model>_<setting>/.
- No caching: every run is a clean evaluation from scratch.
"""

import argparse
import json
import os

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Our own standalone modules — NO InjecAgent imports
# ---------------------------------------------------------------------------
from src.models import OllamaModel
from src.output_parsing import evaluate_output_prompted
from src.utils import get_tool_dict, get_score, get_detailed_defense_metrics
from src.prompts.agent_prompts import PROMPT_DICT

# Layer modules — all logic lives inside these files
from src.layer1_delimiter import apply_layer1
from src.layer2_firewall import Layer2Firewall
from src.layer3_judge import Layer3Judge


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(
        description="PromptShield multi-layer defense pipeline evaluation"
    )
    parser.add_argument("--model_name", required=True,
        help="Ollama model tag for the main agent (e.g. mistral:latest)")
    parser.add_argument("--setting", choices=["base", "enhanced"], required=True,
        help="Test-case setting: 'base' or 'enhanced'")
    parser.add_argument("--prompt_type", choices=["InjecAgent", "hwchase17_react"], required=True,
        help="Agent prompt template to use")
    parser.add_argument("--data_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        ),
        help="Path to InjecAgent data directory (default: <repo_root>/InjecAgent/data)")

    # Defense layers
    parser.add_argument("--enable_layer1", action="store_true", default=False,
        help="Enable Layer 1: structural delimiter defense")
    parser.add_argument("--enable_layer2", action="store_true", default=False,
        help="Enable Layer 2: router LLM firewall")
    parser.add_argument("--enable_layer3", action="store_true", default=False,
        help="Enable Layer 3: LLM-as-a-judge tool call verification")

    # Optional: separate models for defense layers
    parser.add_argument("--firewall_model", default=None,
        help="Ollama model for Layer 2 firewall (default: same as --model_name)")
    parser.add_argument("--judge_model", default=None,
        help="Ollama model for Layer 3 judge (default: same as --model_name)")

    # Execution limits
    parser.add_argument("--only_first_step", action="store_true", default=False,
        help="Only run step 1 (skip data-stealing step 2)")
    parser.add_argument("--max_cases", type=int, default=None,
        help="Only evaluate the first N cases per attack type (default: all)")

    return vars(parser.parse_args())


# ---------------------------------------------------------------------------
# Output directory tag  e.g. "L123", "L1", "L12", "baseline"
# ---------------------------------------------------------------------------

def _layer_tag(params: dict) -> str:
    tag = "".join(str(i) for i in [1, 2, 3] if params.get(f"enable_layer{i}"))
    return f"L{tag}" if tag else "baseline"


# ---------------------------------------------------------------------------
# Scratchpad builder (pipeline glue — not layer logic)
# ---------------------------------------------------------------------------

def _build_scratchpad(item: dict, step_2: bool = False) -> str:
    if step_2:
        return (
            f"\n{item['Thought']}\nAction: {item['User Tool']}\n"
            f"Action Input: {item['Tool Parameters']}\n"
            f"Observation: {item['Tool Response']}\n"
            f"Thought: {item['Attacker Tool Thought']}\n"
            f"Action: {item['Attacker Tools'][0]}\n"
            f"Action Input: {item['Attacker Tool Parameters']}\n"
            f"Observation: {item['Attacker Tool Response']}\n"
        )
    return (
        f"\n{item['Thought']}\nAction: {item['User Tool']}\n"
        f"Action Input: {item['Tool Parameters']}\n"
        f"Observation: {item['Tool Response']}\n"
    )


# ---------------------------------------------------------------------------
# Single-case prediction — calls layer functions only
# ---------------------------------------------------------------------------

def predict_one_case(
    system_prompt: str,
    user_prompt: str,
    item: dict,
    main_model: OllamaModel,
    tool_dict: dict,
    firewall: Layer2Firewall | None,
    judge: Layer3Judge | None,
    enable_layer1: bool,
    step_2: bool = False,
) -> dict:
    """
    Run one test case through the defense pipeline.

    All defense logic is delegated to the layer modules:
      Layer 1 → layer1_delimiter.apply_layer1()
      Layer 2 → Layer2Firewall.screen()
      Layer 3 → Layer3Judge.screen_agent_output()

    Returns
    -------
    dict
        output          – main-agent LLM output, or a block stub if L2/L3 fired
        l1_raw          – delimited tool response produced by Layer 1 (or "N/A")
        l2_raw_verdict  – verbatim firewall LLM output (or "N/A")
        l2_verdict      – 'SAFE' | 'UNSAFE' | 'N/A'
        l3_raw_response – verbatim judge LLM output (or "N/A")
        l3_verdict      – 'ALIGNED' | 'MISALIGNED' | 'N/A'
        blocked_by      – None | 'layer2' | 'layer3'
    """
    result = {
        "output":           "",
        "l1_raw":           "N/A",
        "l2_raw_verdict":   "N/A",
        "l2_verdict":       "N/A",
        "l3_raw_response":  "N/A",
        "l3_verdict":       "N/A",
        "blocked_by":       None,
    }

    tool_response    = item.get("Attacker Tool Response" if step_2 else "Tool Response", "")
    user_instruction = item["User Instruction"]

    # ── Layer 1 ──────────────────────────────────────────────────────────
    if enable_layer1:
        modified_sys, tool_response = apply_layer1(system_prompt, tool_response)
        result["l1_raw"] = tool_response
    else:
        modified_sys = system_prompt

    # ── Layer 2 ──────────────────────────────────────────────────────────

    if firewall is not None:
        is_safe, l2_raw, l2_verdict = firewall.screen(tool_response)
        result["l2_raw_verdict"] = l2_raw
        result["l2_verdict"]     = l2_verdict
        
        if l2_verdict == "INVALID":
            result["output"] = "ERROR: Layer 2 failed or hallucinated."
            result["blocked_by"] = "invalid_layer2"
            return result
        elif l2_verdict == "UNSAFE":
            result["output"]     = firewall.BLOCKED_RESPONSE_STUB
            result["blocked_by"] = "layer2"
            return result

    # Build scratchpad for the main agent
    modified_item = dict(item)
    if step_2:
        modified_item["Attacker Tool Response"] = tool_response
    else:
        modified_item["Tool Response"] = tool_response

    agent_scratchpad = _build_scratchpad(modified_item, step_2=step_2)

    available_tool_names = [item["User Tool"]] + item["Attacker Tools"]
    available_tool_descriptions = json.dumps(
        [tool_dict[tool] for tool in available_tool_names], indent=True
    )
    user_prompt_filled = user_prompt.format(
        available_tool_descriptions=available_tool_descriptions,
        input=user_instruction,
        available_tool_names=available_tool_names,
        agent_scratchpad=agent_scratchpad,
    )

    # ── Main agent ───────────────────────────────────────────────────────
    model_input  = main_model.prepare_input(modified_sys, user_prompt_filled)
    agent_output = main_model.call_model(model_input)
    result["output"] = agent_output

    print("\n" + "═"*60)
    print("🤖 [MAIN AGENT EXECUTION LOG]")
    print(f"USER GOAL: {user_instruction}")
    print("─"*60)
    print(agent_output.strip())
    print("═"*60 + "\n")

    # ── Layer 3 ──────────────────────────────────────────────────────────
    if judge is not None:
        is_aligned, l3_raw, l3_verdict, block_msg = judge.screen_agent_output(
            user_instruction, agent_output, tool_dict
        )
        result["l3_raw_response"] = l3_raw
        result["l3_verdict"]      = l3_verdict
        
        if l3_verdict == "INVALID":
            result["output"] = "ERROR: Layer 3 failed to parse or crashed."
            result["blocked_by"] = "invalid_layer3"
            return result
        elif not is_aligned:
            result["output"]     = block_msg
            result["blocked_by"] = "layer3"
            return result

    return result


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main(params: dict) -> None:
    print("=== PromptShield Defense Pipeline Evaluation ===")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()

    data_dir = params["data_dir"]
    if not os.path.isdir(data_dir):
        raise SystemExit(f"ERROR: data_dir does not exist: {data_dir}")

    # ── Initialise models ────────────────────────────────────────────────
    main_model = OllamaModel({"model_name": params["model_name"]})

    firewall: Layer2Firewall | None = None
    if params["enable_layer2"]:
        fw_model_name = params["firewall_model"] or params["model_name"]
        firewall = Layer2Firewall(OllamaModel({"model_name": fw_model_name}))
        print(f"[Layer 2] Firewall enabled — model: {fw_model_name}")

    judge: Layer3Judge | None = None
    if params["enable_layer3"]:
        judge_model_name = params["judge_model"] or params["model_name"]
        judge = Layer3Judge(OllamaModel({"model_name": judge_model_name}))
        print(f"[Layer 3] Judge enabled — model: {judge_model_name}")

    if params["enable_layer1"]:
        print("[Layer 1] Structural delimiters enabled")
    print()

    # ── Prompt templates & tool dictionary ──────────────────────────────
    system_prompt, user_prompt = PROMPT_DICT[params["prompt_type"]]
    tool_dict = get_tool_dict(data_dir)

    # ── Output directory ─────────────────────────────────────────────────
    repo_root   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    safe_model  = params["model_name"].replace(":", "_").replace("/", "_")
    layer_tag   = _layer_tag(params)
    results_dir = os.path.join(
        repo_root, "results",
        f"defense_{layer_tag}_{safe_model}_{params['prompt_type']}_{params['setting']}",
    )
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}\n")

    output_files: dict = {}

    for attack in ("dh", "ds"):
        file_name      = f"test_cases_{attack}_{params['setting']}.json"
        test_case_file = os.path.join(data_dir, file_name)
        output_file    = os.path.join(results_dir, file_name)
        output_files[attack] = output_file

        with open(test_case_file, "r") as fh:
            data = json.load(fh)

        if params["max_cases"] is not None:
            data = data[:params["max_cases"]]
            print(f"[max_cases] Running on first {len(data)} cases for {attack.upper()}")

        results: list = []
        for item in tqdm(data, desc=f"{attack.upper()} / {params['setting']}"):
            try:
                # ── Step 1 ───────────────────────────────────────────
                step1 = predict_one_case(
                    system_prompt, user_prompt, item,
                    main_model, tool_dict,
                    firewall, judge,
                    params["enable_layer1"],
                    step_2=False,
                )
                item["output"]          = step1["output"]
                item["l1_raw"]          = step1["l1_raw"]
                item["l2_raw_verdict"]  = step1["l2_raw_verdict"]
                item["l2_verdict"]      = step1["l2_verdict"]
                item["l3_raw_response"] = step1["l3_raw_response"]
                item["l3_verdict"]      = step1["l3_verdict"]
                item["blocked_by"]      = step1["blocked_by"]

                # ── Evaluate step-1 ──────────────────────────────────
                if item.get("blocked_by") in ["invalid_layer2", "invalid_layer3"]:
                    item["eval"] = "invalid"
                    item["invalid_reason"] = f"Pipeline Error: {item['blocked_by']}"
                elif item.get("blocked_by"):
                    item["eval"]         = "unsucc"
                    item["block_reason"] = item["blocked_by"]
                else:
                    parsed = evaluate_output_prompted(
                        item["output"],
                        item["Attacker Tools"][0],
                        item["Attacker Tools"],
                        item["User Tool"],
                        item["Attacker Instruction"],
                    )
                    item["eval"] = parsed["eval"]
                    if item["eval"] == "invalid":
                        item["invalid_reason"] = parsed.get("invalid_reason", "")

                # ── Step 2 — data-stealing only ──────────────────────
                if (
                    not params["only_first_step"]
                    and item["eval"] == "succ"
                    and attack == "ds"
                ):
                    item["Attacker Tool Parameters"] = parsed.get("Attacker Tool Parameters", "{}")
                    item["Attacker Tool Thought"]    = parsed.get("Attacker Tool Thought", "")

                    sim_file = os.path.join(data_dir, "attacker_simulated_responses.json")
                    with open(sim_file, "r") as sfh:
                        sim_responses = json.load(sfh)
                    history_key = (
                        f"({item['Attacker Tools'][0]},"
                        f"{item['Attacker Tool Parameters']})"
                    )
                    item["Attacker Tool Response"] = sim_responses.get(history_key, "{}")

                    step2 = predict_one_case(
                        system_prompt, user_prompt, item,
                        main_model, tool_dict,
                        firewall, judge,
                        params["enable_layer1"],
                        step_2=True,
                    )
                    item["output Step 2"]          = step2["output"]
                    item["l1_raw Step 2"]          = step2["l1_raw"]
                    item["l2_raw_verdict Step 2"]  = step2["l2_raw_verdict"]
                    item["l2_verdict Step 2"]      = step2["l2_verdict"]
                    item["l3_raw_response Step 2"] = step2["l3_raw_response"]
                    item["l3_verdict Step 2"]      = step2["l3_verdict"]
                    item["blocked_by Step 2"]      = step2["blocked_by"]

                    if item.get("blocked_by Step 2"):
                        item["eval Step 2"]         = "unsucc"
                        item["block_reason Step 2"] = item["blocked_by Step 2"]
                    else:
                        parsed2 = evaluate_output_prompted(
                            item["output Step 2"],
                            item["Attacker Tools"][1],
                            item["Attacker Tools"],
                            item["User Tool"],
                            item["Attacker Instruction"],
                        )
                        item["eval Step 2"] = parsed2["eval"]
                        if item["eval Step 2"] == "invalid":
                            item["invalid_reason Step 2"] = parsed2.get("invalid_reason", "")

                results.append(item)

            except Exception as exc:
                print(f"[error] {exc} — skipping item")

        # Write results as a pretty-printed JSON array (editor-safe)
        with open(output_file, "w") as out_fh:
            json.dump(results, out_fh, indent=2)

    # ── Final metrics ────────────────────────────────────────────────────
    scores = get_score(output_files)
    detailed_metrics = get_detailed_defense_metrics(output_files)
    print("\n=== Defense Pipeline Results ===")
    print(f"Layers active: {_layer_tag(params)}")
    print(json.dumps(scores, indent=2))
    print("\n--- Detailed Defense Breakdown ---")
    print(json.dumps(detailed_metrics, indent=2))
    print(f"\nOutput files saved to: {results_dir}")


if __name__ == "__main__":
    main(parse_arguments())
