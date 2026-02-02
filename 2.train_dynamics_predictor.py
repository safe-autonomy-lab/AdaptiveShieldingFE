import argparse
import logging
import os

import torch

from FunctionEncoder.Callbacks.LoggerCallback import LoggerCallback
from FunctionEncoder.Dataset.TransitionDataset import TransitionDataset
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder
from shield.dynamics_model.config import AdaptConfig
from shield.dynamics_model.evaluation import adapt_and_eval_oracle, adapt_and_eval_pem, adapt_and_eval_transformer
from shield.dynamics_model.oracle import OracleMLP
from shield.dynamics_model.pem import PEM
from shield.dynamics_model.transformer import TransformerDynamics
from shield.util import load_data, save_config, load_model

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--env_id", type=str, default="SafetyPointGoal1-v1")
    parser.add_argument("--dynamics_model", type=str, choices=["fe","transformer","pem","oracle"], default="fe")
    parser.add_argument("--n_basis", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for finetune/eval")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_dataloader", action="store_true", default=False)  # dataset gives all tensors already
    parser.add_argument("--prediction_horizon", type=int, default=1)
    parser.add_argument("--history_len", type=int, default=5)
    args = parser.parse_args()

    env_name = args.env_id
    env_info = env_name.split('-')[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # ----- Load transitions -----
    train_transitions = load_data(env_info, 'train', prediction_horizon=args.prediction_horizon)
    eval_transitions  = load_data(env_info, 'eval', prediction_horizon=args.prediction_horizon)
    train_X, train_Y  = train_transitions['X'], train_transitions['Y']
    eval_X,  eval_Y   = eval_transitions['X'],  eval_transitions['Y']
    n_functions = len(list(train_X.keys()))

    # Create TransitionDataset: it yields Example/Query tensors with required shapes
    dataset = TransitionDataset(train_transitions, eval_transitions,
                                n_functions=n_functions,
                                n_examples=100, n_queries=900,
                                dtype=torch.float32, device=device)

    example_xs, example_ys, query_xs, query_ys, info = dataset.sample(phase='eval')
    hidden_parameter_dims = len(info['eval_hidden_parameters'][0])
    logger.info("Example_xs %s", example_xs.shape)
    logger.info("Example_ys %s", example_ys.shape)
    logger.info("Query_xs %s", query_xs.shape)
    logger.info("Query_ys %s", query_ys.shape)
    logger.info("Number of hidden parameters %s", len(info['eval_hidden_parameters']))
    
    save_folder = f"saved_files/dynamics_predictor/{env_name}/h{args.prediction_horizon}"
    os.makedirs(save_folder, exist_ok=True)
    save_path = f"{save_folder}/{args.dynamics_model}_model_seed{args.seed}.pth"

    # ----- Build model configs -----
    input_size  = dataset.input_size  # e.g., 14
    output_size = dataset.output_size # e.g., 2

    function_encoder_config = {
        "input_size": input_size, # this is tuple due to function encoder's structure
        "output_size": output_size, # this is tuple due to function encoder's structure
        "data_type": dataset.data_type,
        "n_basis": int(args.n_basis),
        "model_type": "MLP",
        "method": "least_squares",
        "use_residuals_method": True,
        "model_kwargs": {},
        "device": device
    }

    transformer_config = dict(
        input_size=input_size[0],
        output_size=output_size[0],
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        history_len=args.history_len
    )
    pem_config = dict(
        input_size=input_size[0],
        output_size=output_size[0],
        ens_size=5,
        hidden=256,
        layers=3
    )
    oracle_config = dict(
        input_size=input_size[0] + hidden_parameter_dims,   # if you want oracle φ concatenated, add its dim to input_size in your loader
        output_size=output_size[0],
        hidden=256,
        layers=3
    )

    # Save config (useful for reproducibility)
    save_config(function_encoder_config, f"{save_folder}/config.yaml")

    # ----- Create & train or load -----
    if not args.load_model:
        if args.dynamics_model == 'fe':
            model = FunctionEncoder(**function_encoder_config).to(device)
        elif args.dynamics_model == 'transformer':
            model = TransformerDynamics(**transformer_config).to(device)
        elif args.dynamics_model == 'pem':
            model = PEM(**pem_config).to(device)
        elif args.dynamics_model == 'oracle':
            model = OracleMLP(**oracle_config).to(device)
        else:
            raise ValueError("Invalid dynamics model")

    # Provide a light callback that can run eval during training
    cb = LoggerCallback(model, dataset, logdir=f"logs/dynamics_predictor/{env_name}_{args.dynamics_model}_h{args.prediction_horizon}_seed{args.seed}")
    model.train_model(dataset, epochs=int(args.epochs), batch_size=args.batch_size, callback=cb, save_folder=save_folder)
    model.save(save_path)
    logger.info("Saved to %s", save_path)

    exit()

    # ----- Evaluation protocol (few-shot per-context) -----
    # We call our AdaptConfig + adapt_* functions
    adapt_cfg = AdaptConfig(
        lr=2e-3,
        weight_decay=0.0,
        epochs=10,
        batch_size=args.batch_size,
        history_len=args.history_len,
        nll_weight=1.0
    )

    # sample eval tensors
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample(phase='eval')

    if args.dynamics_model == 'oracle':
        metrics = adapt_and_eval_oracle(model, example_xs, example_ys, query_xs, query_ys, adapt_cfg, oracle_feats=None)
    elif args.dynamics_model == 'pem':
        metrics = adapt_and_eval_pem(model, example_xs, example_ys, query_xs, query_ys, adapt_cfg)
    elif args.dynamics_model == 'transformer':
        metrics = adapt_and_eval_transformer(model, example_xs, example_ys, query_xs, query_ys, adapt_cfg)
    elif args.dynamics_model == 'fe':
        # FE typically exposes compute_representation+predict; here we just print shapes as requested
        # Replace this with your FE eval (e.g., eval_fe) if desired.
        logger.info("FE model loaded/trained; use your FE eval path.")
        metrics = {}
    else:
        metrics = {}

    if metrics:
        logger.info("Evaluation: %s", metrics)
