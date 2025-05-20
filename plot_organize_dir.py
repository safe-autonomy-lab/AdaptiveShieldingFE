import os
import shutil
import re

from pathlib import Path
import re
import shutil
from typing import Tuple

def parse_original_path(path: Path) -> Tuple[str, str, int]:
    pattern = r".*?(\d+)_(USL|FAC)_(Safety[A-Za-z]+\d+)_v(\d+)-seed(\d+)-\d+"
    match = re.match(pattern, str(path))
    
    if not match:
        raise ValueError(f"Invalid path format: {path}")
    
    exp_num = match.group(1)
    alg = match.group(2)
    env_name = match.group(3)
    env_version = match.group(4)
    seed = int(match.group(5))
    
    # Construct environment name in the new format
    env = f"{env_name}-v1"
    
    return alg, env, seed

def skit_reorganize_directories(source_dir: str = "./logs/oracle", target_dir: str = "./results/oracle") -> None:
    """
    Reorganize experiment directories from old format to new format.
    
    Args:
        source_dir: Source directory containing original experiment folders
        target_dir: Target directory for reorganized structure
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning directory: {source_path}")
    
    # Process each experiment directory
    for exp_dir in source_path.iterdir():
        if not exp_dir.is_dir():
            continue
            
        try:
            # Parse the directory name
            alg, env, seed = parse_original_path(exp_dir)
            if 'no_oracle' not in source_dir:
                alg = 'Oracle' + alg
            
            # Construct new path
            new_path = target_path / env / alg / f"seed{seed}"
            print(f"Moving {exp_dir.name} to {new_path}")
            
            # Create necessary directories
            new_path.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for file in exp_dir.glob("*"):
                shutil.copy2(file, new_path)
                print(f"  Copied {file.name}")
                
        except ValueError as e:
            print(f"Skipping {exp_dir}: {e}")
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")

def omnisafe_reorganize_runs_directory():
    """
    Reorganize files and include shield parameters in the algorithm name.
    New format for oracle: './results/oracle/SafetyPointGoal0-v1/Shield_s{sampling_nbr}_b{safety_bonus}/seed1'
    New format for non-oracle: './results/standard/SafetyPointGoal0-v1/Shield_s{sampling_nbr}_b{safety_bonus}/seed1'
    """
    import json
    print('Warning! Current base parent dir is wandb_downloads')
    base_dir = "./wandb_downloads"
    pattern = r"([A-Za-z]+)-{([A-Za-z0-9-]+)}-seed-?(\d+)"
    
    # Process both oracle and non-oracle directories
    for run_type in ["oracle_runs", "standard_runs", "fe_runs"]:
        run_dir = os.path.join(base_dir, run_type)
        if not os.path.exists(run_dir):
            continue

        for item in os.listdir(run_dir):
            full_path = os.path.join(run_dir, item)
            
            if not os.path.isdir(full_path):
                continue
                
            match = re.match(pattern, item)
            if match:
                algo_name = match.group(1)
                env_name = match.group(2)
                seed_num = int(match.group(3))
                
                # Read config file to get shield parameters
                config_file = os.path.join(full_path, "config.json")
                if os.path.exists(config_file):
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    
                    if "Shielded" in algo_name:
                        shield_cfg = config["shield_cfgs"]
                        sampling_nbr = shield_cfg.get("sampling_nbr", "NA")
                        safety_bonus = shield_cfg.get("safety_bonus", "NA")
                        static_threshold = shield_cfg.get("static_threshold", "NA")
                        prediction_horizon = shield_cfg.get("prediction_horizon", "NA")
                        penalty_type = shield_cfg.get("penalty_type", "NA")
                        if "use_acp" not in shield_cfg:
                            use_acp = True
                        else:
                            use_acp = shield_cfg.get("use_acp")

                        # this means that we do not use shield
                        if int(prediction_horizon) == 0:
                            if 'TRPO' in algo_name:
                                algo_name = f"TRPOSafetyObjOnly"
                            else:
                                algo_name = f"PPOSafetyObjOnly"
                        # we use shield, but may or may not ACP
                        elif penalty_type == "shield" and use_acp:
                            if 'TRPO' in algo_name:
                                algo_name = f"TRPOShieldACP"
                            else:
                                algo_name = f"PPOShieldACP"
                        else:
                            if 'TRPO' in algo_name:
                                algo_name = f"TRPOShield_s{sampling_nbr}_b{safety_bonus}_t{static_threshold}"
                            else:
                                algo_name = f"PPOShield_s{sampling_nbr}_b{safety_bonus}_t{static_threshold}"
                        
                        if not use_acp:
                            algo_name = "noACP_" + algo_name

                    else:
                        if 'Lag' in algo_name:
                            lag_init = config['lagrange_cfgs']['lagrangian_multiplier_init']
                            lag_lr = config['lagrange_cfgs']['lambda_lr']
                            algo_name = algo_name + f"_init{lag_init}_lr{lag_lr}"
                        else:
                            algo_name = algo_name
                    
                # Create new directory structure including oracle/non-oracle distinction
    
                if run_type == "oracle_runs":
                    results_type = "oracle"
                    algo_name = 'Oracle' + algo_name
                elif run_type == "standard_runs":
                    results_type = "standard"
                elif run_type == "fe_runs":
                    results_type = "fe"
                    algo_name = 'FE' + algo_name

                new_base_dir = os.path.join('./results', results_type, env_name)
                new_algo_dir = os.path.join(new_base_dir, algo_name)
                new_seed_dir = os.path.join(new_algo_dir, f"seed{seed_num}")
                
                # Create all necessary directories
                os.makedirs(new_seed_dir, exist_ok=True)
                
                
                print(f"Moving contents from {full_path} to {new_seed_dir}")
                for file_name in os.listdir(full_path):
                    src_file = os.path.join(full_path, file_name)
                    dst_file = os.path.join(new_seed_dir, file_name)
                    if os.path.exists(dst_file):
                        print(f"Warning: {dst_file} already exists, skipping this file...")
                        continue
                    shutil.copy2(src_file, new_seed_dir)
                
                print(f"Successfully copied to {results_type}/{env_name}/{algo_name}/seed{seed_num}")
                    
def dynamics_reorganize_runs_directory():
    """
    Reorganize files from format './runs/PPOLag-{SafetyPointGoal0-v1}-seed-001-..' 
    to './runs/SafetyPointGoal0-v1/PPOLag/seed1'

    Reorganize files from format './runs/PPOLag-{SafetyPointGoal0-v1}-seed-002-..' 
    to './runs/SafetyPointGoal0-v1/PPOLag/seed2'

    Reorganize files from format './logs/oracle/1_USL_SafetyCarButton1_v1-seed1-0' 
    to './runs/SafetyCarButton1-v1/USL'
    """
    # Base directory
    base_dir = "./logger"
    
    # Updated pattern to also capture seed number
    pattern = r"([A-Za-z]+)_(pem|oracle|transformer|fe)_?(\d+)"
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        
        if not os.path.isdir(full_path):
            continue
            
        match = re.match(pattern, item)
        if match:
            env_name = match.group(1)
            algo_name = match.group(2)
            seed_num = int(match.group(3))
            
            # Create new directory structure including seed
            new_base_dir = os.path.join('./logger', env_name)
            new_algo_dir = os.path.join(new_base_dir, algo_name)
            new_seed_dir = os.path.join(new_algo_dir, f"seed{seed_num}")
            
            # Create all necessary directories
            os.makedirs(new_seed_dir, exist_ok=True)
            
            try:
                # Instead of skipping, copy contents if directory exists
                print(f"Moving contents from {full_path} to {new_seed_dir}")
                for file_name in os.listdir(full_path):
                    src_file = os.path.join(full_path, file_name)
                    dst_file = os.path.join(new_seed_dir, file_name)
                    if os.path.exists(dst_file):
                        print(f"Warning: {dst_file} already exists, skipping this file...")
                        continue
                    shutil.copy2(src_file, new_seed_dir)
                
                print(f"Successfully copied {algo_name} to {env_name}/{algo_name}/seed{seed_num}")
                
            except Exception as e:
                print(f"Error processing {full_path}: {str(e)}")


if __name__ == "__main__":        
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "omni":
        omnisafe_reorganize_runs_directory()
    elif len(sys.argv) > 1 and sys.argv[1] == "skit":
        skit_reorganize_directories()
    elif len(sys.argv) > 1 and sys.argv[1] == "dynamics":
        dynamics_reorganize_runs_directory()
    else:
        print("Usage: python organize_dir.py [omni|skit|dynamics]")
