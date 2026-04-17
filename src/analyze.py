import argparse
import os
import json
from src.utils import get_score, get_detailed_defense_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    # Ensure it's a directory
    if not os.path.isdir(args.path):
        exit(f"Error: {args.path} is not a directory")

    # Map files based on keywords in their names
    files = {}
    for f in os.listdir(args.path):
        full_path = os.path.join(args.path, f)
        if "test_cases_dh" in f: files["dh"] = full_path
        if "test_cases_ds" in f: files["ds"] = full_path

    # Error if both aren't there
    if "dh" not in files or "ds" not in files:
        exit("Error: Folder must contain both 'test_cases_dh' and 'test_cases_ds' files")
    
    if "defense" in files:
        # Run your existing utils functions
        scores = get_score(files)
        detailed = get_detailed_defense_metrics(files)

        print("\nBENCHMARK SCORES:")
        print(json.dumps(scores, indent=2))

        print("\nDETAILED LAYER BREAKDOWN:")
        print(json.dumps(detailed, indent=2))
    else:
        scores = get_score(files)

        print("\nBENCHMARK SCORES:")
        print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()