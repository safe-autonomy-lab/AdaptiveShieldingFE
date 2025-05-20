"""Script to load and evaluate a saved TRPO model from OmniSafe."""
import os
import sys
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import envs
from omnisafe.models import ActorCritic
from omnisafe.models.actor_critic.constraint_actor_q_and_v_critic import ConstraintActorQAndVCritic
from omnisafe.utils.config import Config, ModelConfig
from configuration import EnvironmentConfig
from envs.hidden_parameter_env import HiddenParamEnvs
from omnisafe.envs.wrapper import Normalizer, ObsNormalize
from omnisafe.shield.vectorized_shield import VectorizedShield
from configuration import DynamicPredictorConfig
from omnisafe.evaluator_with_shield import Evaluator


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def create_model_config(config_dict: Dict) -> ModelConfig:
    """Create a ModelConfig object from a configuration dictionary.
    
    Args:
        config_dict: The configuration dictionary
        
    Returns:
        The model configuration object
    """
    model_cfg = ModelConfig()
    model_cfg.actor = Config()
    model_cfg.critic = Config()
    
    # Set actor configuration
    model_cfg.actor.hidden_sizes = config_dict["model_cfgs"]["actor"]["hidden_sizes"]
    model_cfg.actor.activation = config_dict["model_cfgs"]["actor"]["activation"]
    model_cfg.actor.lr = config_dict["model_cfgs"]["actor"]["lr"]
    
    # Set critic configuration
    model_cfg.critic.hidden_sizes = config_dict["model_cfgs"]["critic"]["hidden_sizes"]
    model_cfg.critic.activation = config_dict["model_cfgs"]["critic"]["activation"]
    model_cfg.critic.lr = config_dict["model_cfgs"]["critic"]["lr"]
    
    # Set other model configurations
    model_cfg.weight_initialization_mode = config_dict["model_cfgs"]["weight_initialization_mode"]
    model_cfg.actor_type = config_dict["model_cfgs"]["actor_type"]
    model_cfg.linear_lr_decay = config_dict["model_cfgs"]["linear_lr_decay"]
    model_cfg.exploration_noise_anneal = config_dict["model_cfgs"]["exploration_noise_anneal"]
    model_cfg.std_range = config_dict["model_cfgs"]["std_range"]
    
    return model_cfg

def load_model(model_path: str, config: Dict, env) -> Tuple[ActorCritic, Dict]:
    """Load the saved TRPO model and its configuration.
    
    Args:
        model_path: Path to the saved model weights
        config: The model configuration dictionary
        env: The environment instance
        
    Returns:
        Tuple containing the loaded model and its configuration
    """
    # Load the saved model weights with CPU
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # for key, we ahve 'pi' and 'obs normalizer'
    policy_state_dict = checkpoint["pi"]
    
    # Create model configuration
    model_cfg = create_model_config(config)
    
    # Create model with the same architecture
    model = ConstraintActorQAndVCritic(
        obs_space=env.observation_space,
        act_space=env.action_space,
        model_cfgs=model_cfg,
        epochs=config["train_cfgs"]["epochs"],
    ).to('cpu')  # Explicitly move to CPU
    
    # Load the weights into the model
    model.actor.load_state_dict(policy_state_dict)    
    # Set up observation normalization if available
    if "obs_normalizer" in checkpoint:
        normalizer = Normalizer(env.observation_space.shape)
        # training steps 2_000_000
        normalizer.count = torch.tensor(2_000_000).to('cpu')
        normalizer.load_state_dict(checkpoint["obs_normalizer"])
        env = ObsNormalize(env, norm=normalizer, device="cpu")

    model.eval()
    return model, env, config

def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(env_id: str, algorithm: str, seed: int, sampling_nbr: int):
    """Main function to load and evaluate the model."""
    env_info = env_id.split('-')[0]
    save_env_info = env_info

    # for saved model calling, we need to change the env_info to the one
    env_info = env_info[:-1] + '1'    
    model_path = f"./saved_models/{env_info}/{algorithm}/seed{seed}/torch_save/epoch-100.pt"
    config_path = f"./saved_models/{env_info}/{algorithm}/seed{seed}/config.json"
    # to get the same result
    set_seed(0)
    config = load_config(config_path)
    # render_mode = 'human'
    env_config = EnvironmentConfig()
    # This will trigger out of distribution range sampling [0.1, 0.25] or [1.75, 2.5]
    env_config.MIN_MULT = -1
    env_config.MAX_MULT = -1

    if 'Circle' in env_id:
        env_config.SPEED_LIMIT = False
        env_config.GENERAL_ENV = False
        env_config.CIRCLE_ENV = True
    else:
        env_config.RANGE_LIMIT = -1.0
        env_config.SPEED_LIMIT = False
        env_config.GENERAL_ENV = True
        env_config.CIRCLE_ENV = False
    
    prediction_horizon = 1 if 'Shield' in algorithm else 0
    use_acp = False if 'no_ACP' in algorithm else True

    # For baselines, we use oracle representation
    if 'Shield' not in algorithm and 'SafetyObjOnly' not in algorithm:
        env_config.ORACLE = True
    
    unwrapped_env = HiddenParamEnvs(env_id, device="cpu", env_config=env_config, num_envs=1, render_mode='rgb_array')    
    obs, info = unwrapped_env.reset()

    if 'sigwalls_loc' in info:
        env_config.RANGE_LIMIT = info['sigwalls_loc'][0] - 0.1

    dp_cfgs = DynamicPredictorConfig()
    
    # Load the model and config
    model, env, config = load_model(model_path, config, unwrapped_env)
    if 'Shielded' in algorithm:
        shield = VectorizedShield(env_info=env_info, dynamic_predictor_cfgs=dp_cfgs, env_config=env_config,\
                                sampling_nbr=sampling_nbr, prediction_horizon=prediction_horizon, vector_env_nums=1, use_acp=bool(use_acp))
        shield._setup_environment_params()
        shield.reset()
    else:
        shield = None

    evaluator = Evaluator(env=env, unwrapped_env=unwrapped_env, actor=model.actor, shield=shield)
    evaluator.load_saved(save_dir=f"./saved_models/{env_info}/{algorithm}/seed{seed}", render_mode="rgb_array", env=env)
    episode_rewards, episode_costs, episode_lengths, shield_trigger_counts, episode_run_times, episode_hidden_parameters = evaluator.evaluate(num_episodes=100, cost_criteria=1.0, save_plot=False, seed=0)
    
    # Define the output directory and file path
    output_dir = f"./ood_evaluation_folder/{save_env_info}/{algorithm}_{sampling_nbr}_{prediction_horizon}/seed{seed}"
    output_file_path = os.path.join(output_dir, "evaluation_results.csv")
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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
    # env_id = "SafetyPointGoal1-v1"  # You can change this to match your environment
    baselines_algorithms = ['PPOLag', 'TRPOLag', 'CPO', 'USL']
    our_algorithms = ['ShieldedPPO', 'SafetyObjOnly_PPO', 'ShieldACPPPO', 'ShieldedTRPO', 'SafetyObjOnly_TRPO', 'ShieldACPTRPO']
    algorithms = baselines_algorithms + our_algorithms

    env_id = sys.argv[1]
    algorithm = sys.argv[2]
    assert algorithm in algorithms, f"Algorithm {algorithm} not in {algorithms}"
    seed = int(sys.argv[3])
    sampling_nbr = int(sys.argv[4])
    main(env_id, algorithm, seed, sampling_nbr) 