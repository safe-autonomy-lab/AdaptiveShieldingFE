import os
import pickle
import jax
import jax.numpy as jnp
from typing import Union
from omnisafe.shield.dynamic_predictor import FunctionEncoder, create_train_state, GPFunctionEncoder
from omnisafe.shield.dynamic_predictor.orcacle import OrcaleMLP, oracle_create_train_state
from omnisafe.shield.dynamic_predictor.pem import ProbabilisticEnsembleModel, create_train_state_pem
from omnisafe.shield.dynamic_predictor.transformer import TransformerDynamicPredictor, transformer_create_train_state
from omnisafe.shield.dataset import TransitionDataset
from stable_baselines3.common.logger import configure
from omnisafe.shield.run_utils.dataset_util import create_mo_training_dataset
from configuration import DynamicPredictorConfig

# load transitions from saved files
def load_transitions(env_info, default_path='./saved_files/env_transitions'):
    train_filename = f"{env_info}_train_transitions.pkl"
    eval_filename = f"{env_info}_eval_transitions.pkl"
    train_path = os.path.join(default_path, train_filename)
    eval_path = os.path.join(default_path, eval_filename)

    with open(train_path, 'rb') as f:
        train_transitions = pickle.load(f)
        print(f"Training transitions Loaded from {train_path}")

    with open(eval_path, 'rb') as f:
        eval_transitions = pickle.load(f)
        print(f"Evaluation transitions Loaded from {eval_path}")
    
    return train_transitions, eval_transitions

def process_transitions(transitions, maximum_hidden_parameters: int = 200):
    hiddens = list(transitions.keys())[:maximum_hidden_parameters]
    # hiddens = sorted(hiddens)
    train_in = []
    train_target = []
    for hidden in hiddens:
        train_state_action = []
        train_next_position = []
        for (x, y) in transitions[hidden]:
            train_state_action.append(x)
            train_next_position.append(y)
        
        train_in.append(jnp.array(train_state_action)[None])
        train_target.append(jnp.array(train_next_position)[None])
    
    return jnp.concatenate(train_in, axis=0), jnp.concatenate(train_target, axis=0), list(hiddens)

def call_models(model_name: str, dataset, config: Union[DynamicPredictorConfig], learning_domain: str = 'ds', seed: int = 0, env_info: str = None):
    """Call the models."""
    assert env_info is not None
    assert model_name in ['pem', 'oracle', 'transformer', 'fe', 'gp_fe']
    logger = configure(os.path.join('./logger', env_info + '_' + model_name + '_' + str(seed)), ['csv'])
    rng = jax.random.PRNGKey(seed)
    LEARNING_RATE = config.LEARNING_RATE
    INPUT_DIM = dataset.input_size[0]
    OUTPUT_DIM = dataset.output_size[0]
    HIDDEN_SIZE = config.HIDDEN_SIZE
    HISTORY_LENGTH = config.MAX_HISTORY
    use_attn_encoder = True if learning_domain == 'ts' else False
    
    if model_name == 'pem':
        ENSEMBLE_SIZE = config.ENSEMBLE_SIZE
        pem = ProbabilisticEnsembleModel(input_size=INPUT_DIM, output_size=OUTPUT_DIM, ensemble_size=ENSEMBLE_SIZE, hidden_size=HIDDEN_SIZE, history_length=HISTORY_LENGTH, attn_encoder=use_attn_encoder)
        pem_state = create_train_state_pem(rng, pem, LEARNING_RATE, INPUT_DIM, OUTPUT_DIM, ENSEMBLE_SIZE, learning_domain)
        pem.set_state(pem_state)
        pem.set_dataset(dataset)
        pem.set_train_step()
        return pem, logger
    
    if model_name == 'oracle':
        _, _, _, _, eval_hiddens = dataset.sample(mode='eval', add_hidden_params=True)
        hidden_dim = len(eval_hiddens['hiddens'][0].flatten())
        oracle = OrcaleMLP(input_size=INPUT_DIM + hidden_dim, output_size=OUTPUT_DIM, hidden_size=HIDDEN_SIZE, history_length=HISTORY_LENGTH, attn_encoder=use_attn_encoder)
        oracle_state = oracle_create_train_state(rng, oracle, LEARNING_RATE, INPUT_DIM + hidden_dim, OUTPUT_DIM, learning_domain)
        oracle.set_state(oracle_state)
        oracle.set_dataset(dataset)
        oracle.set_train_step()
        return oracle, logger
    
    if model_name == 'transformer':
        MAX_LEN = config.MAX_LEN
        # for time series update, it takes too much time for transformer, so we make everything half
        if learning_domain == 'ts':
            INPUT_DIM = dataset.input_size[0] * HISTORY_LENGTH
            HIDDEN_SIZE = HIDDEN_SIZE // 2
            d_model = 16
            nhead = 2
            num_encoder_layers = 1
            num_decoder_layers = 1  
        else:
            d_model = 64
            nhead = 4
            num_encoder_layers = 2
            num_decoder_layers = 2
        
        transformer = TransformerDynamicPredictor(input_size=INPUT_DIM, output_size=OUTPUT_DIM, max_len=MAX_LEN, hidden_size=HIDDEN_SIZE, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        state = transformer_create_train_state(rng, transformer, LEARNING_RATE, INPUT_DIM, MAX_LEN)
        transformer.set_state(state)
        transformer.set_dataset(dataset)
        transformer.set_train_step()
        return transformer, logger
    
    if model_name in ['fe', 'gp_fe']:
        N_BASIS = config.N_BASIS
        AVERAGE_FUNCTION = config.AVERAGE_FUNCTION
        LEAST_SQUARES = config.LEAST_SQUARES
        if model_name == 'gp_fe':
            function_encoder = GPFunctionEncoder(INPUT_DIM, OUTPUT_DIM, activation='relu', n_basis=N_BASIS, hidden_size=HIDDEN_SIZE, least_squares=LEAST_SQUARES, average_function=AVERAGE_FUNCTION, history_length=HISTORY_LENGTH, use_attention=use_attn_encoder)
        else:
            function_encoder = FunctionEncoder(INPUT_DIM, OUTPUT_DIM, activation='relu', n_basis=N_BASIS, hidden_size=HIDDEN_SIZE, least_squares=LEAST_SQUARES, average_function=AVERAGE_FUNCTION, history_length=HISTORY_LENGTH, use_attention=use_attn_encoder)
        state = create_train_state(rng, function_encoder, learning_rate=LEARNING_RATE, input_size=INPUT_DIM, output_size=OUTPUT_DIM, learning_domain=learning_domain)
        function_encoder.set_state(state)
        function_encoder.set_dataset(dataset)
        function_encoder.set_train_step()
        return function_encoder, logger
    
def call_transition_dataset(env_info: str, maximum_hidden_parameters: int = 200, nbr_of_examples_per_sample: int = 100, default_path: str = './saved_files/env_transitions'):
    # This is the case for training
    if 'only' not in default_path:
        train_transitions, eval_transitions = load_transitions(env_info, default_path)
    # This is case for evaluation, and varying only one context
    else:
        filename = f"{env_info}_test_transitions.pkl"
        path = os.path.join(default_path, filename)

        with open(path, 'rb') as f:
            train_transitions = pickle.load(f)

        eval_transitions = train_transitions
        print(f"Evaluation transitions Loaded from {path}")
        print(f"Training transitions is the same as evaluation transitions")

    train_in, train_target, hiddens = process_transitions(train_transitions, maximum_hidden_parameters)
    eval_in, eval_target, eval_hiddens = process_transitions(eval_transitions, maximum_hidden_parameters)

    dataset = TransitionDataset(train_in, train_target, hiddens, eval_in, eval_target, eval_hiddens, n_examples_per_sample=nbr_of_examples_per_sample)
    return dataset