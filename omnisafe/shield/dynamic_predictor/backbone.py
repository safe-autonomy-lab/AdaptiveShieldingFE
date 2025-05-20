from typing import Optional
from flax import linen as nn
import flax.serialization
import jax.numpy as jnp
import copy
import os
import pickle
from typing import Any, Dict
from omnisafe.shield.dataset.experience_dataset import ExperienceDataset


class Backbone(nn.Module):
    history_length = None
    input_size = None
    output_size = None
    n_basis = None
    hidden_size = None
    use_attention = None
    activation = None
    least_squares = None
    average_function = None
    """
    A backbone module for dynamic prediction tasks.
    """
    def set_dataset(self, dataset: Any) -> None:
        """
        Set the dataset for the backbone.

        Args:
            dataset (Any): The dataset to be used.
        """
        self.dataset = copy.deepcopy(dataset)

    def get_dataset(self) -> Any:
        """
        Get the current dataset.

        Returns:
            Any: The current dataset.
        """
        return self.dataset
    
    def append_data(self, X: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray) -> None:
        """
        Append new data to the dataset.

        Args:
            X (jnp.ndarray): Input features.
            y (jnp.ndarray): Target values.
            w (jnp.ndarray): Hidden parameters.
        """
        self.dataset.append(X, y, w)

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        set the current state.

        Args:
            state (Dict[str, Any]): The state to be stored.
        """
        self.state = state

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.

        Returns:
            Dict[str, Any]: The current state.
        """
        return self.state

    def save_train_state(self, filename: str, default_path: str = '.') -> None:
        """Save both model state and configuration."""
        print('Saving state...')
        os.makedirs(default_path, exist_ok=True)
        
        # Save both state and config
        save_dict = {
            'state': flax.serialization.to_bytes(self.state),
            'config': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'n_basis': self.n_basis,
                'hidden_size': self.hidden_size,
                'history_length': self.history_length,
                'use_attention': self.use_attention,
                'activation': self.activation,
                'least_squares': self.least_squares,
                'average_function': self.average_function
            }
        }
        
        with open(os.path.join(default_path, filename), 'wb') as f:
            pickle.dump(save_dict, f)
        print('State and config saved successfully')

    def load_train_state(self, filename: str, state_template: Dict[str, Any], default_path: str = '.') -> None:
        """Load both model state and verify configuration matches."""
        print('Loading state...')
        
        with open(os.path.join(default_path, filename), 'rb') as f:
            save_dict = pickle.load(f)
        
        # Verify configuration matches
        saved_config = save_dict['config']
        current_config = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'n_basis': self.n_basis,
            'hidden_size': self.hidden_size,
            'history_length': self.history_length,
            'use_attention': self.use_attention,
            'activation': self.activation,
            'least_squares': self.least_squares,
            'average_function': self.average_function
        }
        
        # Check for mismatches
        for key in saved_config:
            if saved_config[key] != current_config[key]:
                raise ValueError(f"Configuration mismatch for {key}: saved={saved_config[key]}, current={current_config[key]}")
        
        # Load state
        loaded_state = flax.serialization.from_bytes(state_template, save_dict['state'])
        print('State loaded successfully')
        self.state = loaded_state
        
    def set_episode_data_memory(self):
        # obs, acs, next_obs
        self.episode_data_memory = []
    
    def add_episode_data_memory(self, data):
        self.episode_data_memory.append(data)

    def set_experience_dataset(self, experience_dataset: Optional[ExperienceDataset] = None):
        """
        Set the experience dataset.

        Args:
            experience_dataset (Optional[ExperienceDataset], optional): The experience dataset to be used. Defaults to None.
        """
        self.experience_dataset = experience_dataset
        if experience_dataset is not None:
            self.experience_dataset = ExperienceDataset(input_size=self.input_size, output_size=self.output_size, hidden_param_size=self.hidden_param_size)

    def get_experience_dataset(self) -> ExperienceDataset:
        return self.experience_dataset
    
    def get_train_step(self):
        return self.train_step

    def set_train_step(self):
        raise NotImplementedError

