from typing import Tuple, Dict
import jax.numpy as jnp

from .base_dataset import BaseDataset


class TransitionDataset(BaseDataset):
    """
    A dataset class for generating transition data using JAX.
    """
    def __init__(
        self,
        transition_states_and_actions: jnp.ndarray,
        transition_next_states: jnp.ndarray,
        hiddens: jnp.ndarray,
        eval_transition_states_and_actions: jnp.ndarray,
        eval_transition_next_states: jnp.ndarray,
        eval_hiddens: jnp.ndarray,
        n_examples_per_sample: int = 100,
        dtype: jnp.dtype = jnp.float32,
    ): 
        
        n_functions_per_sample = transition_states_and_actions.shape[0]
        n_points_per_sample = transition_states_and_actions.shape[1] - n_examples_per_sample
        input_size = transition_states_and_actions.shape[-1]
        output_size = transition_next_states.shape[-1]
        super().__init__(
            input_size=(input_size, ),
            output_size=(output_size, ),
            total_n_functions=float("inf"),
            total_n_samples_per_function=float("inf"),
            n_functions_per_sample=n_functions_per_sample,
            n_examples_per_sample=n_examples_per_sample,
            n_points_per_sample=n_points_per_sample,
            dtype=dtype,
        )
        self.transition_states_and_actions = transition_states_and_actions
        self.transition_next_states = transition_next_states
        self.hiddens = hiddens
        self.eval_transition_states_and_actions = eval_transition_states_and_actions
        self.eval_transition_next_states = eval_transition_next_states
        self.eval_hiddens = eval_hiddens
        self.n_hiddens = len(hiddens[0])

    def sample(self, mode: str = "train", add_hidden_params: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Generate a sample of transitions.

        Returns:
            A tuple containing:
            - states: Input values for example points.
            - actions: Output values for example points.
            - next_states: Input values for evaluation points.
        """
        assert mode in ["train", "eval"], f"Invalid mode: {mode}. Please specify 'train' or 'eval'."
        
        if mode == "train":
            example_xs = self.transition_states_and_actions[:, :self.n_examples_per_sample, :]
            example_ys = self.transition_next_states[:, :self.n_examples_per_sample, :]
            xs = self.transition_states_and_actions[:, self.n_examples_per_sample:, :]
            ys = self.transition_next_states[:, self.n_examples_per_sample:, :]            
            hiddens = self.hiddens
            if add_hidden_params:
                hiddens = jnp.array(hiddens).reshape(-1, 1, self.n_hiddens)
                hidden_repeat = jnp.repeat(hiddens, xs.shape[1], axis=1)
                hidden_repeat_for_example = jnp.repeat(hiddens, example_xs.shape[1], axis=1)
                example_xs = jnp.concatenate([example_xs, hidden_repeat_for_example], axis=-1)
                xs = jnp.concatenate([xs, hidden_repeat], axis=-1)   

        elif mode == "eval":
            example_xs = self.eval_transition_states_and_actions[:, :self.n_examples_per_sample, :]
            example_ys = self.eval_transition_next_states[:, :self.n_examples_per_sample, :]
            xs = self.eval_transition_states_and_actions[:, self.n_examples_per_sample:, :]
            ys = self.eval_transition_next_states[:, self.n_examples_per_sample:, :]
            hiddens = self.eval_hiddens
            if add_hidden_params:
                hiddens = jnp.array(hiddens).reshape(-1, 1, self.n_hiddens)
                hidden_repeat = jnp.repeat(hiddens, xs.shape[1], axis=1)
                hidden_repeat_for_example = jnp.repeat(hiddens, example_xs.shape[1], axis=1)
                example_xs = jnp.concatenate([example_xs, hidden_repeat_for_example], axis=-1)
                xs = jnp.concatenate([xs, hidden_repeat], axis=-1)   

        return example_xs, example_ys, xs, ys, {"hiddens": hiddens}