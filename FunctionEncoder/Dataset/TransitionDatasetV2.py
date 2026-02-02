from typing import Tuple, Dict, Any, Optional
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from .BaseDataset import BaseDataset

logger = logging.getLogger(__name__)


class TransitionDatasetV2(Dataset):
    """
    Memory-efficient PyTorch Dataset for transition data using lazy loading.
    Uses torch.utils.data.Dataset interface for automatic DataLoader compatibility.
    """
    
    def __init__(
        self,
        train_transitions: Dict,
        eval_transitions: Dict,
        n_functions: int = 1,
        n_examples: int = 20,
        n_queries: int = 20,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        mode: str = "train"
    ):
        """
        Args:
            train_transitions: Dictionary with 'X' and 'Y' keys containing training data
            eval_transitions: Dictionary with 'X' and 'Y' keys containing evaluation data  
            n_functions: Number of functions per batch (handled by DataLoader)
            n_examples: Number of example points per function
            n_queries: Number of query points per function
            dtype: Data type for tensors
            device: Device to place tensors on
            mode: 'train' or 'eval' mode
        """
        
        # Store transitions as references (not copies)
        self.train_transitions = train_transitions
        self.eval_transitions = eval_transitions
        self.mode = mode
        self.n_examples = n_examples
        self.n_queries = n_queries
        self.dtype = dtype
        self.device = device
        
        # Get dataset info
        train_X = train_transitions['X']
        train_Y = train_transitions['Y']
        
        self.train_hidden_parameters = list(train_X.keys())
        self.eval_hidden_parameters = list(eval_transitions['X'].keys())
        
        # Get data dimensions from first function
        first_key = self.train_hidden_parameters[0]
        self.total_points, self.input_size = np.shape(train_X[first_key])
        _, self.output_size = np.shape(train_Y[first_key])
        
        # Validate we have enough points
        required_points = n_examples + n_queries
        if self.total_points < required_points:
            raise ValueError(f"Not enough data points. Required: {required_points}, Available: {self.total_points}")
        
        # Store function keys for indexing
        if mode == "train":
            self.function_keys = self.train_hidden_parameters
        else:
            self.function_keys = self.eval_hidden_parameters
            
        self.dt = 0.02
        
        logger.info(f"TransitionDatasetV2 created - Mode: {mode}")
        logger.info(f"Functions: {len(self.function_keys)}, Input size: {self.input_size}, Output size: {self.output_size}")
        logger.info(f"Examples: {n_examples}, Queries: {n_queries}, Total points per function: {self.total_points}")

    def __len__(self) -> int:
        """Return number of functions in dataset."""
        return len(self.function_keys)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single function's data by index.
        
        Args:
            idx: Index of the function to retrieve
            
        Returns:
            Tuple of (example_xs, example_ys, query_xs, query_ys, info)
        """
        function_key = self.function_keys[idx]
        
        # Get raw data for this function (still numpy arrays)
        if self.mode == "train":
            X_data = self.train_transitions['X'][function_key]
            Y_data = self.train_transitions['Y'][function_key]
        else:
            X_data = self.eval_transitions['X'][function_key]
            Y_data = self.eval_transitions['Y'][function_key]
        
        # Split into examples and queries
        example_xs = X_data[:self.n_examples]
        example_ys = Y_data[:self.n_examples]
        query_xs = X_data[self.n_examples:self.n_examples + self.n_queries]
        query_ys = Y_data[self.n_examples:self.n_examples + self.n_queries]
        
        # Convert to tensors (this is where memory is allocated per batch)
        example_xs_tensor = torch.tensor(example_xs, dtype=self.dtype, device=self.device)
        example_ys_tensor = torch.tensor(example_ys, dtype=self.dtype, device=self.device)
        query_xs_tensor = torch.tensor(query_xs, dtype=self.dtype, device=self.device)
        query_ys_tensor = torch.tensor(query_ys, dtype=self.dtype, device=self.device)
        
        # Info dict
        info = {f'{self.mode}_hidden_parameters': [function_key]}
        
        return example_xs_tensor, example_ys_tensor, query_xs_tensor, query_ys_tensor, info
    
    def get_input_size(self) -> Tuple[int]:
        """Return input size tuple for compatibility."""
        return (self.input_size,)
    
    def get_output_size(self) -> Tuple[int]:
        """Return output size tuple for compatibility."""
        return (self.output_size,)
    
    def get_data_type(self) -> str:
        """Return data type for compatibility."""
        return "deterministic"


def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Custom collate function to handle batching of function data.
    
    Args:
        batch: List of (example_xs, example_ys, query_xs, query_ys, info) tuples
        
    Returns:
        Batched tensors and combined info dict
    """
    example_xs_list, example_ys_list, query_xs_list, query_ys_list, info_list = zip(*batch)
    
    # Stack tensors along batch dimension
    example_xs = torch.stack(example_xs_list, dim=0)
    example_ys = torch.stack(example_ys_list, dim=0)
    query_xs = torch.stack(query_xs_list, dim=0)
    query_ys = torch.stack(query_ys_list, dim=0)
    
    # Combine info dicts
    combined_info = {}
    for info in info_list:
        for key, value in info.items():
            if key not in combined_info:
                combined_info[key] = []
            combined_info[key].extend(value)
    
    return example_xs, example_ys, query_xs, query_ys, combined_info
