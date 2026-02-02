"""Script to load and evaluate a saved TRPO model from OmniSafe."""
import os
import sys
import numpy as np
import pandas as pd
import torch
from organize_run_folders import _find_latest_run_dir, _organize_run_artifacts, _find_latest_epoch_file
from envs.safety_gymnasium.configuration import EnvironmentConfig
from envs.hidden_parameter_env import HiddenParamEnvs, VelocityParamEnv
from shield.vectorized_shield import VectorizedShield
from omnisafe.evaluator_with_shield import Evaluator
from load_utils import load_model, load_config, set_seed
from shield.util import load_model as load_model_shield
from envs.safety_gymnasium.utils.registration import make
import logging


def _algo_name_from_penalty(algorithm: str, penalty_type: str) -> str:
    penalty = str(penalty_type or "").lower()
    if penalty == "reward":
        return f"{algorithm}withSRO"
    if penalty == "sro":
        base_algo = algorithm
        if base_algo.startswith("Shielded"):
            base_algo = base_algo[len("Shielded") :]
        return f"{base_algo}withSRO"
    return algorithm


def _find_results_dir(env_id: str, algorithm: str, seed: int, prediction_horizon: int) -> str | None:
    candidates: list[str] = []
    if algorithm.startswith("Shielded"):
        base_algo = algorithm[len("Shielded") :]
        algo_variants = [algorithm, f"{algorithm}withSRO", f"{base_algo}withSRO"]
        for algo in algo_variants:
            candidates.append(os.path.join("results", "fe", env_id, f"h{prediction_horizon}", algo, f"seed{seed}"))
            candidates.append(os.path.join("results", "oracle", env_id, f"h{prediction_horizon}", algo, f"seed{seed}"))
    else:
        candidates.append(os.path.join("results", "oracle", env_id, f"h{prediction_horizon}", f"Oracle-{algorithm}", f"seed{seed}"))
        candidates.append(os.path.join("results", "oracle", env_id, f"h{prediction_horizon}", algorithm, f"seed{seed}"))
    for path in candidates:
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    return None


def main(env_id: str, algorithm: str, seed: int, sampling_nbr: int, prediction_horizon: int, threshold: float, idle_condition: int, scale: float, num_eval_episodes: int, n_basis: int):
    """Main function to load and evaluate the model."""
    shield_logger = logging.getLogger("shield")
    shield_logger.propagate = False
    shield_logger.setLevel(logging.INFO)
    env_info = env_id.split('-')[0]
    
    if 'Shielded' not in algorithm:
        results_dir = f"./results/oracle/{env_id}/h{prediction_horizon}/Oracle-{algorithm}/seed{seed}"
        runs_base = f"./runs/Oracle-{algorithm}-{env_id}"
    else:
        results_dir = f"./results/fe/{env_id}/h{prediction_horizon}/{algorithm}/seed{seed}"
        runs_base = f"./runs/{algorithm}-{env_id}"

    run_dir = _find_latest_run_dir(runs_base, seed)
    if run_dir:
        organized_dir = _organize_run_artifacts(run_dir, env_id, algorithm, seed)
        if organized_dir:
            results_dir = organized_dir
        torch_save_dir = os.path.join(results_dir, "torch_save")
        model_path = _find_latest_epoch_file(torch_save_dir)
        config_path = os.path.join(results_dir, "config.json")
        load_path = results_dir
    else:
        fallback_dir = _find_results_dir(env_id, algorithm, seed, prediction_horizon)
        if fallback_dir:
            results_dir = fallback_dir
        torch_save_dir = os.path.join(results_dir, "torch_save")
        model_path = _find_latest_epoch_file(torch_save_dir)
        config_path = os.path.join(results_dir, "config.json")
        load_path = results_dir
    if model_path is None:
        raise FileNotFoundError(f"No epoch-*.pt found in {torch_save_dir}")
        
    # to get the same result
    set_seed(0)
    device = 'cpu'
    config = load_config(config_path)
    shield_cfgs = config.get("shield_cfgs", {})
    penalty_type = str(shield_cfgs.get("penalty_type", "")).lower()
    algo_name = _algo_name_from_penalty(algorithm, penalty_type)
    eval_env_id = (
        config.get("env_id")
        or config.get("env_name")
        or config.get("train_cfgs", {}).get("env_id")
        or config.get("env_cfgs", {}).get("env_id")
        or env_id
    )
    # render_mode = 'human'
    env_config = EnvironmentConfig()
    env_config.IS_OUT_OF_DISTRIBUTION = True
    env_config.FIX_HIDDEN_PARAMETERS = False
    env_config.NBR_OF_BASIS = n_basis
    if 'Shielded' not in algorithm:
        env_config.USE_ORACLE = True

    # If environment is not supported by HiddenParamEnvs, it means training was done without it (so no N_BASIS dims were added)
    
    if 'Velocity' in eval_env_id:
        env_config.MIN_MULT = 0.7
        env_config.MAX_MULT = 1.3
        unwrapped_env = VelocityParamEnv(eval_env_id, device=device, env_config=env_config, num_envs=1, render_mode='rgb_array')
    else:
        unwrapped_env = HiddenParamEnvs(eval_env_id, device=device, env_config=env_config, num_envs=1, render_mode='rgb_array')

    
    # Load the model and config
    model, env, config = load_model(model_path, config, unwrapped_env, device=device)
    if 'Shielded' in algorithm and penalty_type != "sro":
        dynamics_model_type = shield_cfgs.get("dynamics_model", "fe")
        dynamic_model = None
        # Determine base environment name for dynamics predictor
        # e.g. SafetyPointGoal2-v1 -> SafetyPointGoal1-v1 (assuming using v1 model for v2 task) or specific logic
        # For now, just using env_info as base path component if needed, or keeping existing logic if correct for project structure
        # The original code did: env_info[:-1] + '1-v1' which implies replacing '2' with '1'.
        # Let's keep the path logic consistent with the project's 'saved_files' structure.
        
        dynamics_path = f'saved_files/dynamics_predictor/{eval_env_id}'
        
        if dynamic_model is None:
            dynamics_path = f'saved_files/dynamics_predictor/{eval_env_id}'
            dynamic_model = load_model_shield(
                dynamics_path,
                dynamics_model_type,
                device=device,
                prediction_horizon=prediction_horizon,
                seed=seed,
            )
        shield = VectorizedShield(
            dynamic_predictor=dynamic_model,
            mo_predictor=None,
            sampling_nbr=sampling_nbr,
            prediction_horizon=prediction_horizon,
            vector_env_nums=1,
            static_threshold=threshold,
            example_nbr=100,
            idle_condition=idle_condition,
            dynamics_model_type=dynamics_model_type,
            scale=scale,
        )
        shield.reset()
        
        # Inject environment properties manually since we are using unwrapped_env/Eval wrapper
        shield.is_circle = 'Circle' in env_id
        shield.is_velocity = 'Velocity' in env_id
        shield.is_swimmer = 'Swimmer' in env_id
        shield.is_cheetah = 'Cheetah' in env_id
        if shield.is_circle and hasattr(unwrapped_env, 'sigwalls_loc'):
             # This might be tricky if unwrapped_env is a vector wrapper or hidden param wrapper
             # Assuming standard access pattern or default
             pass 

        evaluator = Evaluator(env=env, unwrapped_env=unwrapped_env, actor=model, shield=shield, safety_budget=config.get('safety_budget', 1.0))
    else:
        shield = None
        evaluator = Evaluator(env=env, unwrapped_env=unwrapped_env, actor=model.actor, shield=None, safety_budget=config.get('safety_budget', 1.0))
    
    evaluator.load_saved(save_dir=load_path, render_mode="rgb_array", env=env)
    
    # Construct output directory
    if 'Shielded' in algorithm and penalty_type != "sro":
        suffix = f"_s{sampling_nbr}_th{threshold}_idle{idle_condition}_h{prediction_horizon}_scale{scale}"
        dir_name = f"{algo_name}{suffix}"
    elif 'Shielded' in algorithm and penalty_type == "sro":
        dir_name = algo_name
    else:
        dir_name = algorithm
        
    output_dir = f"./ood_evaluation_folder/{env_info}/{dir_name}/seed{seed}"
    
    output_file_path = os.path.join(output_dir, "evaluation_results.csv")
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")
    episode_rewards, episode_costs, episode_lengths, shield_trigger_counts, episode_run_times, episode_hidden_parameters = evaluator.evaluate(num_episodes=num_eval_episodes, cost_criteria=1.0, save_plot=False, seed=seed)
    
    # Save episode rewards and hidden parameters using numpy's .npz format
    data_file_path = os.path.join(output_dir, "episode_data.npz")
    np.savez(
        data_file_path,
        rewards=np.array(episode_rewards),
        hidden_parameters=np.array(episode_hidden_parameters),
        costs=np.array(episode_costs),
        lengths=np.array(episode_lengths),
        shield_trigger_counts=np.array(shield_trigger_counts),
        run_times=np.array(episode_run_times)
    )
        
    # Prepare data for DataFrame
    results_data = {
        "Metric": [
            "Average episode reward",
            "Average episode cost",
            "Average episode length",
            "Average shield triggered",
            "Average episode run time"
        ],
        "Value": [
            np.mean(a=episode_rewards),
            np.mean(a=episode_costs),
            np.mean(a=episode_lengths),
            np.mean(a=shield_trigger_counts),
            np.mean(a=episode_run_times)
        ]
    }
    results_df = pd.DataFrame(results_data)

    # Save the DataFrame to CSV
    results_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    # env_id = "SafetyPointGoal2-v1"  # You can change this to match your environment
    # python 4.load_model.py SafetyCarCircle2-v1 ShieldedRCPO_s50_b1.0_i4 0 5 1 0.2 4
    env_id = sys.argv[1]
    algorithm = sys.argv[2]
    seed = int(sys.argv[3])
    sampling_nbr = int(sys.argv[4])
    prediction_horizon = int(sys.argv[5])
    threshold = float(sys.argv[6]) # this is the lipschitz constant of the dynamics predictor
    idle_condition = int(sys.argv[7]) # this control engagement of the shield
    scale = float(sys.argv[8]) # this controls the diversity of actions sampled from the policy
    num_eval_episodes = int(sys.argv[9])
    n_basis = int(sys.argv[10])
    main(env_id, algorithm, seed, sampling_nbr, prediction_horizon, threshold, idle_condition, scale, num_eval_episodes, n_basis) 
