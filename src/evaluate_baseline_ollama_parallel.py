#!/usr/bin/env python3
"""
Parallel evaluation runner - processes multiple test cases concurrently.

Uses ThreadPoolExecutor for I/O-bound operations with thread-safe queue.
Safe for Ollama API calls which are I/O bound.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
import threading

from tqdm import tqdm

# Local imports
from src.models import OllamaModel
from src.output_parsing import evaluate_output_prompted
from src.utils import get_tool_dict, get_score
from src.prompts.agent_prompts import PROMPT_DICT


# Thread-safe lock for file writing
write_lock = threading.Lock()


def predict_one_case(
    system_prompt: str,
    user_prompt: str,
    item: dict,
    model: OllamaModel,
    tool_dict: dict,
    step_2: bool = False,
) -> str:
    """Single prediction (same as original)."""
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


def evaluate_test_case(
    item: dict,
    system_prompt: str,
    user_prompt: str,
    model: OllamaModel,
    tool_dict: dict,
    data_dir: str,
    attack_type: str,
    only_first_step: bool = False,
) -> Tuple[dict, Exception]:
    """
    Evaluate a single test case (may include step 1 and step 2).
    Returns (modified_item, exception_if_any)
    """
    try:
        # Step 1
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

        # Step 2 (data-stealing only, only for DS attacks)
        if (
            not only_first_step
            and item["eval"] == "succ"
            and attack_type == "ds"
        ):
            item["Attacker Tool Parameters"] = parsed["Attacker Tool Parameters"]
            item["Attacker Tool Thought"] = parsed["Attacker Tool Thought"]

            sim_file = os.path.join(data_dir, "attacker_simulated_responses.json")
            with open(sim_file, "r") as sfh:
                sim_responses = json.load(sfh)
            history_key = (
                f"({item['Attacker Tools'][0]},{item['Attacker Tool Parameters']})"
            )
            item["Attacker Tool Response"] = sim_responses.get(history_key, "{}")

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

        return item, None

    except Exception as exc:
        return item, exc


def main(params: dict) -> None:
    print("=== Parallel Baseline Ollama Evaluation ===")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()

    data_dir = params["data_dir"]
    if not os.path.isdir(data_dir):
        raise SystemExit(f"ERROR: data_dir does not exist: {data_dir}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    safe_model = params["model_name"].replace(":", "_").replace("/", "_")
    results_dir = os.path.join(
        repo_root,
        "results",
        f"baseline_Ollama_{safe_model}_{params['prompt_type']}_{params['setting']}",
    )
    os.makedirs(results_dir, exist_ok=True)

    # Initialize model (shared across threads)
    model = OllamaModel({"model_name": params["model_name"]})

    system_prompt, user_prompt = PROMPT_DICT[params["prompt_type"]]
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

        print(f"\nProcessing {attack.upper()} ({len(data)} cases)")
        print(f"  Parallel workers: {params['num_workers']}")

        # Load cache if resuming
        existing_results: dict = {}
        if params["use_cache"] and os.path.exists(output_file):
            with open(output_file, "r") as fh:
                for line in fh:
                    try:
                        tmp = json.loads(line)
                        key = (tmp["User Instruction"], tmp["Tool Response"])
                        existing_results[key] = tmp
                    except Exception:
                        continue
            print(f"[cache] loaded {len(existing_results)} results from {output_file}")

        # Prepare items to evaluate (skip cached ones)
        items_to_eval = []
        for item in data:
            key = (item["User Instruction"], item["Tool Response"])
            if params["use_cache"] and key in existing_results:
                # Already evaluated, skip
                continue
            items_to_eval.append(item)

        # Parallel evaluation
        with ThreadPoolExecutor(max_workers=params["num_workers"]) as executor:
            futures = {
                executor.submit(
                    evaluate_test_case,
                    item,
                    system_prompt,
                    user_prompt,
                    model,
                    tool_dict,
                    data_dir,
                    attack,
                    params["only_first_step"],
                ): idx
                for idx, item in enumerate(items_to_eval)
            }

            # Write results as they complete
            with open(output_file, "w") as out_fh:
                # First, write all cached results
                for key, cached_item in existing_results.items():
                    out_fh.write(json.dumps(cached_item) + "\n")

                # Then write new results as they complete
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{attack.upper()} / {params['setting']}",
                ):
                    item, exc = future.result()
                    if exc:
                        print(f"[error] {exc} — skipping item")
                    else:
                        with write_lock:
                            out_fh.write(json.dumps(item) + "\n")

    # Final metrics
    scores = get_score(output_files)
    print("\n=== Results ===")
    print(json.dumps(scores, indent=2))
    print(f"\nOutput files saved to: {results_dir}")


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(
        description="Parallel baseline evaluation — local Ollama model"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Ollama model tag, e.g. mistral:latest",
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
        help="Agent prompt template",
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        ),
        help="Path to test-case JSON files (default: <repo_root>/data)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker threads (default: 4). "
             "Increase for faster GPU, decrease if OOM or Ollama bottleneck.",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Only evaluate first N cases per attack type",
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
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(parse_arguments())
