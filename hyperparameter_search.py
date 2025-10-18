import random
import subprocess
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import itertools

# Import the search space configuration
import config

def generate_random_hparam_configs(search_space, num_trials):
    """Generates a list of unique, random hyperparameter combinations."""
    keys = search_space.keys()
    value_lists = [search_space[key] for key in keys]
    all_permutations = list(itertools.product(*value_lists))
    random.shuffle(all_permutations)
    num_to_run = min(num_trials, len(all_permutations))
    selected_trials = all_permutations[:num_to_run]
    hparam_configs = [dict(zip(keys, trial)) for trial in selected_trials]
    return hparam_configs

def main():
    """
    Main function to run the hyperparameter search.
    """
    search_results_dir = Path("./hyperparam_search_results")
    search_results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = search_results_dir / "hparam_search_summary.csv"

    hparam_configs = generate_random_hparam_configs(config.HPARAM_SEARCH_SPACE, config.NUM_SEARCH_TRIALS)
    total_trials = len(hparam_configs)

    print(f"--- Starting Hyperparameter Search for {total_trials} random trials ---")

    if summary_csv_path.exists():
        results_df = pd.read_csv(summary_csv_path)
    else:
        columns = list(config.HPARAM_SEARCH_SPACE.keys()) + ["best_val_recall", "best_val_precision", "best_val_f1", "model_path"]
        results_df = pd.DataFrame(columns=columns)

    for i, params in enumerate(hparam_configs):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n--- Starting Trial {i+1}/{total_trials} (Run ID: {run_id}) ---")
        print(f"Parameters: {params}")

        # --- NEW: Create descriptive filenames for model and detailed log ---
        # Create a compact string from the current hyperparameter combination for the filename
        params_str = "_".join([f"{key.replace('_','')[0:2]}{value}" for key, value in params.items()])

        model_filename = f"model_{run_id}_{params_str}.pth"
        log_filename = f"log_{run_id}_{params_str}.csv"

        model_save_path = search_results_dir / model_filename
        results_csv_path = search_results_dir / log_filename # Path for the detailed log
        # --- END NEW ---

        command = [
            'python', 'train_model.py',
            '--model-save-path', str(model_save_path),
            '--results-csv-path', str(results_csv_path) # Pass the new log path
        ]
        for key, value in params.items():
            command.extend([f'--{key}', str(value)])

        try:
            proc = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
            run_results = json.loads(proc.stdout)
            log_entry = {**params, **run_results}

            print(f"--- Trial {i+1} Finished ---")
            print(f"Results: R={run_results['best_val_recall']:.4f}, P={run_results['best_val_precision']:.4f}, F1={run_results['best_val_f1']:.4f}")
            print(f"Detailed epoch log saved to: {results_csv_path}")

        except subprocess.CalledProcessError as e:
            print(f"!!! Trial {i+1} FAILED !!!")
            print(f"Error output:\n{e.stderr}")
            log_entry = {**params, "best_val_recall": "ERROR", "best_val_precision": "ERROR", "best_val_f1": "ERROR", "model_path": e.stderr.strip()}

        except json.JSONDecodeError:
            print(f"!!! Trial {i+1} FAILED !!!")
            print("Could not decode JSON from train_model.py output. Check for errors.")
            log_entry = {**params, "best_val_recall": "ERROR", "best_val_precision": "ERROR", "best_val_f1": "ERROR", "model_path": "JSON DECODE ERROR"}


        results_df = pd.concat([results_df, pd.DataFrame([log_entry])], ignore_index=True)
        results_df.to_csv(summary_csv_path, index=False)
        print(f"Summary logged to {summary_csv_path}")

    print("\n--- Hyperparameter Search Finished ---")
    # Make sure F1 score column is numeric before sorting
    results_df['best_val_f1'] = pd.to_numeric(results_df['best_val_f1'], errors='coerce')
    sorted_results = results_df.sort_values(by="best_val_f1", ascending=False)
    print("Top 5 performing parameter sets:")
    print(sorted_results.head(5).to_string())

if __name__ == "__main__":
    main()

