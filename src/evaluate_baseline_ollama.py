"""
evaluate_baseline_ollama.py
===========================
Standalone baseline evaluation of the InjecAgent benchmark using a
local Ollama model.

Zero code from InjecAgent is imported — this file and its sibling
modules (models.py, output_parsing.py, utils.py, prompts/) are
fully self-contained.

The only thing referenced from InjecAgent/ is the *data* directory
(JSON/JSONL files), which is passed in via --data_dir.

Usage
-----
Run via:
    scripts/run_baseline_ollama.sh

Or manually:
    python3 src/evaluate_baseline_ollama.py \\
        --model_name  mistral:latest \\
        --setting     base \\
        --prompt_type InjecAgent \\
        --data_dir    InjecAgent/data \\
        [--use_cache] [--only_first_step]
"""

import argparse
import json
import os

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports (all from our own src/)
# ---------------------------------------------------------------------------
from src.models import OllamaModel
from src.output_parsing import evaluate_output_prompted
from src.utils import get_tool_dict, get_score
from src.prompts.agent_prompts import PROMPT_DICT


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(
        description="Baseline IPI evaluation — local Ollama model"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Ollama model tag, e.g. mistral:latest or llama3:8b",
    )
    parser.add_argument(
        "--setting",
        choices=["base", "enhanced"],
        required=True,
        help="Test-case setting: 'base' or 'enhanced'",
    )
    parser.add_argument(
        "--prompt_type",
        choices=["InjecAgent", "hwchase17_react"],
        required=True,
        help="Agent prompt template to use",
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        ),
        help="Path to the directory containing InjecAgent test-case JSON files "
             "(default: <repo_root>/data)",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=False,
        help="Resume from an existing partial output file",
    )
    parser.add_argument(
        "--only_first_step",
        action="store_true",
        default=False,
        help="Only run step 1 (skip data-stealing step 2)",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Only evaluate the first N cases per attack type (default: all)",
    )
    return vars(parser.parse_args())


# ---------------------------------------------------------------------------
# Single-case prediction
# ---------------------------------------------------------------------------

def predict_one_case(
    system_prompt: str,
    user_prompt: str,
    item: dict,
    model: OllamaModel,
    tool_dict: dict,
    step_2: bool = False,
) -> str:
    available_tool_names = [item["User Tool"]] + item["Attacker Tools"]
    available_tool_descriptions = json.dumps(
        [tool_dict[tool] for tool in available_tool_names], indent=True
    )
    input_text = item["User Instruction"]

    if step_2:
        agent_scratchpad = (
            f"\n{item['Thought']}\nAction: {item['User Tool']}\n"
            f"Action Input: {item['Tool Parameters']}\n"
            f"Observation: {item['Tool Response']}\n"
            f"Thought: {item['Attacker Tool Thought']}\n"
            f"Action: {item['Attacker Tools'][0]}\n"
            f"Action Input: {item['Attacker Tool Parameters']}\n"
            f"Observation: {item['Attacker Tool Response']}\n"
        )
    else:
        agent_scratchpad = (
            f"\n{item['Thought']}\nAction: {item['User Tool']}\n"
            f"Action Input: {item['Tool Parameters']}\n"
            f"Observation: {item['Tool Response']}\n"
        )

    user_prompt_filled = user_prompt.format(
        available_tool_descriptions=available_tool_descriptions,
        input=input_text,
        available_tool_names=available_tool_names,
        agent_scratchpad=agent_scratchpad,
    )
    model_input = model.prepare_input(system_prompt, user_prompt_filled)
    return model.call_model(model_input)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main(params: dict) -> None:
    print("=== Baseline Ollama Evaluation ===")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()

    data_dir = params["data_dir"]
    if not os.path.isdir(data_dir):
        raise SystemExit(f"ERROR: data_dir does not exist: {data_dir}")

    # Results directory (project-level results/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    safe_model = params["model_name"].replace(":", "_").replace("/", "_")
    results_dir = os.path.join(
        repo_root,
        "results",
        f"baseline_Ollama_{safe_model}_{params['prompt_type']}_{params['setting']}",
    )
    os.makedirs(results_dir, exist_ok=True)

    # Initialise Ollama model
    model = OllamaModel({"model_name": params["model_name"]})

    # Prompt templates
    system_prompt, user_prompt = PROMPT_DICT[params["prompt_type"]]

    # Tool dictionary loaded from data_dir
    tool_dict = get_tool_dict(data_dir)

    output_files: dict = {}

    for attack in ("dh", "ds"):
        file_name = f"test_cases_{attack}_{params['setting']}.json"
        test_case_file = os.path.join(data_dir, file_name)
        output_file = os.path.join(results_dir, file_name)
        output_files[attack] = output_file

        with open(test_case_file, "r") as fh:
            data = json.load(fh)

        if params["max_cases"] is not None:
            data = data[:params["max_cases"]]
            print(f"[max_cases] Running on first {len(data)} cases for {attack.upper()}")

        # ------------------------------------------------------------------
        # Cache: load any previously computed outputs to allow resuming
        # ------------------------------------------------------------------
        existing_step1: dict = {}
        existing_step2: dict = {}
        if params["use_cache"] and os.path.exists(output_file):
            with open(output_file, "r") as fh:
                for line in fh:
                    try:
                        tmp = json.loads(line)
                        if isinstance(tmp.get("output"), str) and tmp["output"]:
                            existing_step1[
                                (tmp["User Instruction"], tmp["Tool Response"])
                            ] = tmp["output"]
                        if isinstance(tmp.get("output Step 2"), str) and tmp["output Step 2"]:
                            existing_step2[
                                (
                                    tmp["User Instruction"],
                                    tmp["Tool Response"],
                                    tmp["output"],
                                )
                            ] = tmp["output Step 2"]
                    except Exception:
                        continue
            print(
                f"[cache] loaded {len(existing_step1)} step-1 "
                f"and {len(existing_step2)} step-2 outputs from {output_file}"
            )

        # ------------------------------------------------------------------
        # Evaluation loop
        # ------------------------------------------------------------------
        results: list = []
        for item in tqdm(data, desc=f"{attack.upper()} / {params['setting']}"):
            try:
                # Step 1 ------------------------------------------------
                key1 = (item["User Instruction"], item["Tool Response"])
                if params["use_cache"] and key1 in existing_step1:
                    output = existing_step1[key1]
                else:
                    output = predict_one_case(
                        system_prompt, user_prompt, item, model, tool_dict
                    )
                item["output"] = output

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

                # Step 2 (data-stealing only) ----------------------------
                if (
                    not params["only_first_step"]
                    and item["eval"] == "succ"
                    and attack == "ds"
                ):
                    item["Attacker Tool Parameters"] = parsed["Attacker Tool Parameters"]
                    item["Attacker Tool Thought"] = parsed["Attacker Tool Thought"]

                    sim_file = os.path.join(data_dir, "attacker_simulated_responses.json")
                    with open(sim_file, "r") as sfh:
                        sim_responses = json.load(sfh)
                    history_key = (
                        f"({item['Attacker Tools'][0]},"
                        f"{item['Attacker Tool Parameters']})"
                    )
                    item["Attacker Tool Response"] = sim_responses.get(history_key, "{}")

                    key2 = (
                        item["User Instruction"],
                        item["Tool Response"],
                        item["output"],
                    )
                    if params["use_cache"] and key2 in existing_step2:
                        output2 = existing_step2[key2]
                    else:
                        output2 = predict_one_case(
                            system_prompt,
                            user_prompt,
                            item,
                            model,
                            tool_dict,
                            step_2=True,
                        )
                    item["output Step 2"] = output2

                    parsed2 = evaluate_output_prompted(
                        output2,
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
                print(f"[error] {exc} — skipping item in {output_file}")

        # Write results as a pretty-printed JSON array (editor-safe)
        with open(output_file, "w") as out_fh:
            json.dump(results, out_fh, indent=2)

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    scores = get_score(output_files)
    print("\n=== Results ===")
    print(json.dumps(scores, indent=2))
    print(f"\nOutput files saved to: {results_dir}")


if __name__ == "__main__":
    main(parse_arguments())
