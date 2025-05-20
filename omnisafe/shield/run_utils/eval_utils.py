from omnisafe.shield.dynamic_predictor import GPFunctionEncoder, OrcaleMLP, ProbabilisticEnsembleModel
import numpy as np

ABS_LOSS = True
NBR_OF_EXAMPLES_PER_SAMPLE = 100

abs_loss_fn = lambda y_hat, y: 2 * np.mean(np.abs(y_hat - y))
square_loss_fn = lambda y_hat, y: np.mean((y_hat - y) ** 2)

def eval_fe(example_eval_in: np.ndarray, example_eval_target: np.ndarray, eval_in: np.ndarray, eval_target: np.ndarray, model: GPFunctionEncoder) -> float:
    """Evaluate function encoder performance."""
    step_loss = 0
    state = model.get_state()
    length_of_eval = len(eval_in)
    for idx in range(length_of_eval):
        idx_slice = slice(idx, idx + 1)
        test_x, test_y = eval_in[idx_slice], eval_target[idx_slice]
        coefficients = model.get_compute_coefficients_fn()(state, example_eval_in[idx_slice], example_eval_target[idx_slice])
        y_hat = model.predict(coefficients, test_x)
        if ABS_LOSS:
            step_loss += abs_loss_fn(y_hat, test_y)
        else:   
            step_loss += square_loss_fn(y_hat, test_y)
    return step_loss / length_of_eval


def eval_transformer(example_eval_in: np.ndarray, example_eval_target: np.ndarray, eval_in: np.ndarray, eval_target: np.ndarray, learning_domain: str, model) -> float:
    """Evaluate transformer model performance."""
    step_loss = 0
    state = model.get_state()
    length_of_eval = len(eval_in)
    for idx in range(length_of_eval):
        example_x, test_y, test_x = example_eval_in[idx], eval_target[idx], eval_in[idx]
        # For timesseries prediction, each moving obstalces are considered as a single function
        if learning_domain == 'ts':
            example_x, test_x, test_y = example_x[:, None, :], test_x[:, None, :], test_y[:, None, :]
        # For dynamic transition prediction, one dynamics is considered as a single function
        elif learning_domain == 'ds':
            example_x, test_x, test_y = example_x[None, :, :], test_x[None, :, :], test_y[None, :, :]
        y_hat = model.predict_difference(state, example_x, test_x)
        if ABS_LOSS:
            step_loss += abs_loss_fn(y_hat, test_y)
        else:   
            step_loss += square_loss_fn(y_hat, test_y)
        
    return step_loss / length_of_eval


def eval_oracle(eval_in: np.ndarray, eval_target: np.ndarray, oracle) -> float:
    """Evaluate oracle model performance."""
    step_loss = 0
    state = oracle.get_state()
    length_of_eval = len(eval_in)
    for idx in range(length_of_eval):
        test_x, test_y = eval_in[idx], eval_target[idx]
        y_hat = OrcaleMLP.predict_difference(state, test_x)
        if ABS_LOSS:
            step_loss += abs_loss_fn(y_hat, test_y)
        else:   
            step_loss += square_loss_fn(y_hat, test_y)
    
    return step_loss / length_of_eval


def eval_pem(eval_in: np.ndarray, eval_target: np.ndarray, pem) -> float:
    """Evaluate probabilistic ensemble model performance."""
    step_loss = 0
    state = pem.get_state()
    length_of_eval = len(eval_in)    
    for idx in range(length_of_eval):
        test_x, test_y = eval_in[idx], eval_target[idx]
        y_hat, _ = ProbabilisticEnsembleModel.predict_difference(state, test_x, ign_var=True)
        if ABS_LOSS:
            step_loss += abs_loss_fn(y_hat, test_y)
        else:   
            step_loss += square_loss_fn(y_hat, test_y)
    return step_loss / length_of_eval
