import wandb
import json
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import sys


# Initialize WandB API
api = wandb.Api()

ENTITY = "Please change this to your wandb project entity"

# ---- Diagnostic code to list projects ----
print(f"Attempting to list projects for entity: {ENTITY}")
try:
    projects = api.projects(entity=ENTITY)
    print(f"Available projects for entity '{ENTITY}':")
    if projects:
        for project in projects:
            print(f"- {project.name}")
    else:
        print("No projects found for this entity.")
except Exception as e:
    print(f"Error listing projects: {e}")
# ---- End of diagnostic code ----

PROJECT = sys.argv[1]

# Get all runs from the project
runs = api.runs(f"{ENTITY}/{PROJECT}")

# Create a directory to store all downloads
output_dir = Path("wandb_downloads")
output_dir.mkdir(exist_ok=True)

# Create separate directories for oracle and non-oracle runs
oracle_dir = output_dir / "oracle_runs"
fe_dir = output_dir / "fe_runs"
oracle_dir.mkdir(exist_ok=True)
fe_dir.mkdir(exist_ok=True)

def download_run_data(run, run_dir):
    """Download history, config, and summary for a run with retry mechanism."""
    # Download history and save as CSV
    history = run.history()
    history_path = run_dir / "history.csv"
    history.to_csv(history_path, index=False)
    print(f"Saved history to {history_path}")
    
    # Download config and save as JSON
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(run.config, f, indent=4)
    print(f"Saved config to {config_path}")
    
    # Download summary and save as JSON
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(run.summary._json_dict, f, indent=4)
    print(f"Saved summary to {summary_path}")

# Download data for each run
successful_downloads = 0
failed_downloads = []
# shield configs
sampling_condition = [5, 10, 50, 100]
safety_bonuse_condition = [1]
run_states = ['finished', 'running']
BASELINE = False
baselines = ['TRPOLag', 'PPOLag', 'CPO']
shields = ['ShieldedPPOLag', 'ShieldedTRPOLag']
# if baseline is true then we only download baseline runs
if BASELINE:
    # algo_list = baselines
    algo_list = ['TRPOLag', "PPOLag"]
else:
    algo_list = shields

env_list = ['SafetyPointButton1', 'SafetyCarButton1', 'SafetyPointCircle1', 'SafetyCarCircle1', 'SafetyCarPush1', 'SafetyPointPush1', 'SafetyPointGoal1', 'SafetyCarGoal1']

# BASELINE = False
for run in tqdm(runs, desc="Processing runs"):
    config = run.config
    algo_name = config['algo']
    parts = run.name.split('-')
    env_name = parts[1][1:]
    seed_nbr = int(parts[4])

    if algo_name == 'TRPOLag' and seed_nbr in [0, 1, 2] and 'PointButton' in env_name:
        print(f"Algo name is TRPOLag and seed nbr is {seed_nbr}")
        
    if len(env_list) > 0 and env_name not in env_list:
        continue

    if algo_name not in algo_list:
        continue

    shield_config = config['shield_cfgs']
    env_config = config['env_cfgs']['env_config']
    prediction_horizon = shield_config['prediction_horizon']
    
    if not BASELINE:
        if (int(shield_config['sampling_nbr']) not in sampling_condition):
            continue
    
    print(f"\nProcessing run: {run.name}")
    
    # Determine the base directory based on oracle configuration
    if env_config['use_oracle']:
        base_dir = oracle_dir
    # this is older runs 
    elif env_config['use_fe_representation']:
        base_dir = fe_dir
    else:
        raise ValueError(f"Unknown representation: {env_config['use_fe_representation']}")

    # Create a directory for this specific run
    run_dir = base_dir / run.name
    run_dir.mkdir(exist_ok=True)
    
    # Download data with retry mechanism
    download_run_data(run, run_dir)
    
    # Add delay between downloads
    time.sleep(0.5)
    
    successful_downloads += 1

print("\nDownload Summary:")
print(f"Successfully downloaded: {successful_downloads} runs")
if failed_downloads:
    print("\nFailed downloads:")
    for run_name, error in failed_downloads:
        print(f"- {run_name}: {error}")
else:
    print("No failed downloads!")