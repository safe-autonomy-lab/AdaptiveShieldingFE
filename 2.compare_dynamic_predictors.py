from dataclasses import dataclass
from typing import Any
from functools import partial
from omnisafe.shield.run_utils.load_utils import call_transition_dataset, call_models
from omnisafe.shield.run_utils.eval_utils import eval_fe, eval_pem, eval_oracle, eval_transformer
from configuration import DynamicPredictorConfig
import os
import jax
import numpy as np
import random
import wandb

@dataclass
class ModelConfig:
    """Configuration for model training."""
    use_fe: bool = True
    use_gp_fe: bool = False
    use_oracle: bool = False
    use_pem: bool = False
    use_transformer: bool = False

def print_dataset_info(dataset: Any, mode: str) -> None:
    """Print dataset shape information."""
    example_in, example_target, eval_in, eval_target, hiddens = dataset.sample(
        mode=mode, 
        add_hidden_params=False
    )
    len_hiddens = len(hiddens['hiddens'])
    print(f"Mode: {mode}")
    print(f"Number of Collected Hidden Parameters: {len_hiddens}")
    print(f"Example input shape: {example_in.shape}")
    print(f"Example target shape: {example_target.shape}")
    print(f"Eval input shape: {eval_in.shape}")
    print(f"Eval target shape: {eval_target.shape}")

def train_model(
    model_type: str,
    config: DynamicPredictorConfig,
    dataset: Any,
    env_info: str,
    add_hidden_params: bool = False,
    model_save: bool = False,
    seed: int = 0,
    use_wandb: int = 0
) -> None:
    """Generic model training function."""
    print(f"Starting training {model_type}")
    # Model initialization
    if use_wandb:
        wandb.login()
        wandb.init(
            project="dynamic-predictor",
            config={"model_name": model_type, "epochs": config.EPOCH}
        )
    model, logger = call_models(model_type, dataset, config, seed=seed, env_info=env_info, learning_domain='ds')
    state = model.get_state()
    model.set_dataset(dataset)
    
    # Evaluation setup
    example_eval_in, example_target, eval_in, eval_target, _ = dataset.sample(
        mode="eval",
        add_hidden_params=add_hidden_params
    )
    
    # Model-specific eval function setup
    eval_funcs = {
        "fe": partial(eval_fe, example_eval_in, example_target, eval_in, eval_target),
        "gp_fe": partial(eval_fe, example_eval_in, example_target, eval_in, eval_target),
        "pem": partial(eval_pem, eval_in, eval_target),
        "oracle": partial(eval_oracle, eval_in, eval_target),
        "transformer": partial(eval_transformer, example_eval_in, example_target, eval_in, eval_target, 'ds')
    }

    if model_type == "fe":
        filename = f"{env_info}_{config.EPOCH}.pkl"
    else:
        filename = f"{env_info}_{config.EPOCH}_{model_type}.pkl"
    # Training
    model.train_model(
        epochs=config.EPOCH,
        logger=logger,
        eval_func=eval_funcs[model_type],
        use_wandb=use_wandb,
    )
    wandb.finish()
    if model_save and model_type == "fe":
        save_path = "./saved_files/trained_dynamic_predictor"    
        os.makedirs(save_path, exist_ok=True)
        model.save_train_state(filename, save_path)
        model.load_train_state(filename, state, save_path)

def set_seeds(seed) -> None:
    """Set seeds for reproducibility across all random number generators.
    
    Args:
        seed: Integer seed value for random number generators
    """
    # Set seeds for different RNG sources
    random.seed(seed)
    np.random.seed(seed)
    # Set both JAX CPU and GPU/TPU seeds
    jax.random.PRNGKey(seed)

def main(env_id: str, seed: int = 42, use_wandb: int = 0) -> None:
    """Main function to run the experiment."""
    # Set seeds before any randomization occurs
    set_seeds(seed)
    
    # Configuration
    config = DynamicPredictorConfig()
    model_config = ModelConfig()
    
    # Environment setup
    env_info = env_id.split("-")[0]
    nbr_of_examples_per_sample = 50 if 'Circle' in env_id else 100
    dataset = call_transition_dataset(env_info, maximum_hidden_parameters=1000, nbr_of_examples_per_sample=nbr_of_examples_per_sample)    
    
    for mode in ["eval", "train"]:
        print_dataset_info(dataset, mode)
        
    print(f"INPUT_DIM: {dataset.input_size[0]}, OUTPUT_DIM: {dataset.output_size[0]}")
    # Train selected models
    model_configs = [
        ("fe", model_config.use_fe, False),
        ("gp_fe", model_config.use_gp_fe, False),
        ("pem", model_config.use_pem, False),
        # only oracle is used add hidden parameters!
        ("oracle", model_config.use_oracle, True),
        ("transformer", model_config.use_transformer, False),
    ]
    
    for model_type, should_train, add_hidden_params in model_configs:
        if should_train:
            train_model(model_type, config, dataset, env_info, add_hidden_params, model_save=True, seed=seed, use_wandb=use_wandb)

if __name__ == "__main__":
    import sys
    env_id = sys.argv[1]
    seed = int(sys.argv[2])
    use_wandb = int(sys.argv[3])
    main(env_id, seed, use_wandb)