from typing import Tuple, Dict, Any
import torch
import numpy as np
import logging
from .BaseDataset import BaseDataset


class TransitionDataset(BaseDataset):
    """
    A dataset class for generating transition data using PyTorch.
    """
    def __init__(
        self,
        train_transitions,
        eval_transitions,
        n_functions:int=1,
        n_examples:int=20,
        n_queries:int=20,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ): 
        train_transitions_X = train_transitions['X']
        train_transitions_Y = train_transitions['Y']
        eval_transitions_X = eval_transitions['X']
        eval_transitions_Y = eval_transitions['Y']

        self.train_hidden_parameters = list(train_transitions_X.keys())
        self.eval_hidden_parameters = list(eval_transitions_X.keys())
        self.n_queries, self.input_size = np.shape(train_transitions_X[self.train_hidden_parameters[0]])
        _, self.output_size = np.shape(train_transitions_Y[self.train_hidden_parameters[0]])
        
        # Keep data as numpy arrays initially - only convert to tensors when needed
        self.train_X = np.stack([train_transitions_X[key] for key in self.train_hidden_parameters], axis=0)
        self.train_Y = np.stack([train_transitions_Y[key] for key in self.train_hidden_parameters], axis=0)
        self.eval_X = np.stack([eval_transitions_X[key] for key in self.eval_hidden_parameters], axis=0)
        self.eval_Y = np.stack([eval_transitions_Y[key] for key in self.eval_hidden_parameters], axis=0)
        self.dt = 0.02
        
        super().__init__(
            input_size=(self.input_size, ),
            output_size=(self.output_size, ),
            total_n_functions=float("inf"),
            total_n_samples_per_function=float("inf"),
            data_type="deterministic",
            n_functions=n_functions,
            n_examples=n_examples,
            n_queries=self.n_queries,
            dtype=dtype,
        )
        self.n_queries = self.n_queries - self.n_examples
        self.device = device
        
        # Add state for batch iteration
        self._current_epoch = 0
        self._batch_start_idx = 0


    def sample(self, phase: str = "train", use_sequential_batching: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate a sample of transitions.

        Args:
            phase: Training phase ('train' or 'eval')
            use_sequential_batching: If True, use sequential batching instead of random sampling

        Returns:
            A tuple containing:
            - states: Input values for example points.
            - actions: Output values for example points.
            - next_states: Input values for evaluation points.
        """
        assert phase in ["train", "eval"], f"Invalid phase: {phase}. Please specify 'train' or 'eval'."

        if phase == "train":
            num_fns = self.train_X.shape[0]
            n_functions = min(num_fns, self.n_functions)
            
            if use_sequential_batching:
                # Sequential batching for better coverage
                start_idx = self._batch_start_idx
                end_idx = min(start_idx + n_functions, num_fns)
                
                # Handle wrapping around
                if end_idx - start_idx < n_functions:
                    # Need to wrap around to beginning
                    indices_1 = list(range(start_idx, num_fns))
                    indices_2 = list(range(0, n_functions - len(indices_1)))
                    fn_indices = np.array(indices_1 + indices_2)
                    self._batch_start_idx = len(indices_2)
                else:
                    fn_indices = np.arange(start_idx, end_idx)
                    self._batch_start_idx = end_idx % num_fns
            else:
                # Random sampling (original behavior)
                fn_indices = np.random.choice(num_fns, n_functions, replace=False)
            
            example_xs = self.train_X[fn_indices, :self.n_examples, ...]
            example_ys = self.train_Y[fn_indices, :self.n_examples, ...]
            xs = self.train_X[fn_indices, self.n_examples:, ...]
            ys = self.train_Y[fn_indices, self.n_examples:, ...]
            info = {'train_hidden_parameters': [self.train_hidden_parameters[i] for i in fn_indices]}

        elif phase == "eval":
            num_fns = self.eval_X.shape[0]
            n_functions = min(num_fns, self.n_functions)
            
            if use_sequential_batching:
                # For eval, we can use simpler sequential approach
                start_idx = self._batch_start_idx
                end_idx = min(start_idx + n_functions, num_fns)
                
                if end_idx - start_idx < n_functions:
                    indices_1 = list(range(start_idx, num_fns))
                    indices_2 = list(range(0, n_functions - len(indices_1)))
                    fn_indices = np.array(indices_1 + indices_2)
                else:
                    fn_indices = np.arange(start_idx, end_idx)
            else:
                fn_indices = np.random.choice(num_fns, n_functions, replace=False)

            example_xs = self.eval_X[fn_indices, :self.n_examples, ...]
            example_ys = self.eval_Y[fn_indices, :self.n_examples, ...]
            xs = self.eval_X[fn_indices, self.n_examples:, ...]
            ys = self.eval_Y[fn_indices, self.n_examples:, ...]
            info = {'eval_hidden_parameters': [self.eval_hidden_parameters[i] for i in fn_indices]}
        else:
            raise ValueError(f"Invalid phase: {phase}. Please specify 'train' or 'eval'.")
        
        # Convert to tensors only when needed and log memory usage
        example_xs_tensor = torch.tensor(example_xs, dtype=self.dtype, device=self.device)
        example_ys_tensor = torch.tensor(example_ys, dtype=self.dtype, device=self.device)
        xs_tensor = torch.tensor(xs, dtype=self.dtype, device=self.device)
        ys_tensor = torch.tensor(ys, dtype=self.dtype, device=self.device)       
        return example_xs_tensor, example_ys_tensor, xs_tensor, ys_tensor, info
    
    def reset_batch_state(self):
        """Reset batch iteration state for new epoch."""
        self._batch_start_idx = 0