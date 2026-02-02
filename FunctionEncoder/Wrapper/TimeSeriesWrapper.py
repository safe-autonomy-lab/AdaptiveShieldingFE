import torch
import yaml
import os
import logging
from typing import Dict, Any
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder
from tqdm import trange

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TimeSeriesWrapper:
    """
    Wrapper around FunctionEncoder to convert delta predictions to absolute position predictions.
    
    The original FunctionEncoder predicts position deltas (changes), but for time series prediction,
    we often want absolute positions. This wrapper accumulates the deltas to provide absolute positions.
    
    Also provides enhanced save/load functionality that includes configuration management.
    
    Args:
        function_encoder (FunctionEncoder): The wrapped FunctionEncoder instance
        feature_dim (int): Dimension of the feature space (e.g., 2 for (x,y) positions)
    """
    
    def __init__(self, function_encoder: FunctionEncoder, feature_dim: int = 2, history_length: int = 10):
        self.function_encoder = function_encoder
        self.feature_dim = feature_dim
        self.history_length = history_length
        self.name = 'function_encoder'

    def compute_representation(self, example_xs: torch.tensor, example_ys: torch.tensor, **kwargs) -> torch.tensor:
        return self.function_encoder.compute_representation(example_xs, example_ys, **kwargs)

    def predict(self, x: torch.tensor, coeffs: torch.tensor, prediction_horizon: int, **kwargs) -> torch.tensor:
        assert prediction_horizon >= 1, f"prediction_horizon must be >= 1, got {prediction_horizon}"
        assert x.shape[-1] == self.feature_dim, \
            f"x feature dimension {x.shape[-1]} doesn't match feature_dim {self.feature_dim}"
        
        if prediction_horizon == 1:
            return self._single_step_prediction(x, coeffs, prediction_horizon, **kwargs)
        else:
            return self._multi_step_prediction(x, coeffs, prediction_horizon, **kwargs)

    def _single_step_prediction(self, 
                               x: torch.tensor, 
                               coeffs: torch.tensor, 
                               **kwargs) -> torch.tensor:
        """
        Perform single-step prediction (original behavior).
        
        Returns:
            torch.tensor: Shape (n_functions, n_queries, feature_dim)
        """
        # Get delta predictions from the underlying FunctionEncoder
        delta_predictions = self.function_encoder.predict(x, coeffs, **kwargs)
        # Convert deltas to absolute positions
        absolute_predictions = self._deltas_to_absolute_positions(x, delta_predictions)
        
        logger.info(f"Single-step prediction completed. Output shape: {absolute_predictions.shape}")
        return absolute_predictions, delta_predictions
    
    def _multi_step_prediction(self, 
                              x: torch.tensor, 
                              coeffs: torch.tensor, 
                              n_steps: int,
                              **kwargs) -> torch.tensor:
        """
        Perform multi-step prediction by iteratively using predicted outputs as inputs.
        
        Args:
            example_xs: Example input trajectories. Shape: (n_functions, n_examples, history_length, feature_dim)
            example_ys: Example delta targets. Shape: (n_functions, n_examples, feature_dim)  
            query_xs: Query input trajectories. Shape: (n_functions, n_queries, history_length, feature_dim)
            method: Prediction method
            n_steps: Number of prediction steps
            **kwargs: Additional arguments
            
        Returns:
            torch.tensor: Multi-step predictions. Shape (n_functions, n_queries, n_steps, feature_dim)
        """
        n_functions, n_queries, history_length, feature_dim = x.shape
        logger.info(f"Multi-step prediction: n_functions={n_functions}, n_queries={n_queries}, "
                   f"history_length={history_length}, feature_dim={feature_dim}")
        
        # Initialize storage for all predictions
        all_predictions = torch.zeros(n_functions, n_queries, n_steps, feature_dim, 
                                    dtype=  x.dtype, device=x.device)
        all_delta_predictions = torch.zeros(n_functions, n_queries, n_steps, feature_dim, 
                                    dtype=x.dtype, device=x.device)
        # Start with the original query sequences
        current_query_xs = x.clone()
        
        for step in range(n_steps):
            logger.info(f"Processing step {step + 1}/{n_steps}")
            
            # Validate current query shape
            assert current_query_xs.shape == (n_functions, n_queries, history_length, feature_dim), \
                f"Step {step}: Unexpected current_query_xs shape: {current_query_xs.shape}"
            
            # Get delta predictions for current step
            delta_predictions = self.function_encoder.predict(current_query_xs, coeffs, **kwargs)
            
            # Convert deltas to absolute positions
            absolute_predictions = self._deltas_to_absolute_positions(current_query_xs, delta_predictions)
            
            # Store predictions for this step
            all_predictions[:, :, step, :] = absolute_predictions
            all_delta_predictions[:, :, step, :] = delta_predictions
            logger.info(f"Step {step + 1} completed. Prediction shape: {absolute_predictions.shape}")
            
            # Update query sequences for next step (if not the last step)
            if step < n_steps - 1:
                current_query_xs = self._update_query_sequences(current_query_xs, absolute_predictions)
        
        logger.info(f"Multi-step prediction completed. Final output shape: {all_predictions.shape}")
        return all_predictions, all_delta_predictions
    
    def _update_query_sequences(self, 
                               current_query_xs: torch.tensor, 
                               new_predictions: torch.tensor) -> torch.tensor:
        """
        Update query sequences by sliding the window and appending new predictions.
        
        This implements the sliding window approach where:
        - We remove the oldest time step from the sequence
        - We append the newly predicted position as the latest time step
        
        Args:
            current_query_xs: Current query sequences. Shape: (n_functions, n_queries, history_length, feature_dim)
            new_predictions: New predicted positions. Shape: (n_functions, n_queries, feature_dim)
            
        Returns:
            torch.tensor: Updated query sequences with same shape as input
        """
        n_functions, n_queries, history_length, feature_dim = current_query_xs.shape
        
        # Validate input shapes
        assert new_predictions.shape == (n_functions, n_queries, feature_dim), \
            f"new_predictions shape {new_predictions.shape} doesn't match expected shape"
        
        # Create new query sequences by sliding the window
        # Take all time steps except the first one: current_query_xs[:, :, 1:, :]
        # Then append the new predictions as the latest time step
        updated_sequences = torch.cat([
            current_query_xs[:, :, 1:, :],  # Remove oldest time step
            new_predictions.unsqueeze(2)     # Add new prediction as latest time step
        ], dim=2)
        
        # Validate output shape
        assert updated_sequences.shape == (n_functions, n_queries, history_length, feature_dim), \
            f"Updated sequences shape {updated_sequences.shape} doesn't match expected shape"
        
        return updated_sequences

    def _deltas_to_absolute_positions(self, query_xs: torch.tensor, delta_predictions: torch.tensor) -> torch.tensor:
        """
        Convert delta predictions to absolute positions.
        
        The key insight: To predict the absolute position at the next time step,
        we take the last position from the input sequence and add the predicted delta.
        
        Args:
            query_xs: Query input trajectories. Shape: (n_functions, n_queries, history_length, feature_dim)
            delta_predictions: Predicted deltas. Shape: (n_functions, n_queries, feature_dim)
            
        Returns:
            torch.tensor: Absolute position predictions. Shape: (n_functions, n_queries, feature_dim)
        """
        # Extract the last position from each query sequence
        # query_xs[:, :, -1, :] gives us the last time step for each query
        last_positions = query_xs[:, :, -1, :]
        
        # Add deltas to get absolute positions: position_next = position_last + delta
        absolute_positions = last_positions + delta_predictions
        return absolute_positions
    
    def train_model(self,
                dataset, 
                epochs: int,
                progress_bar=True,
                callback=None,
                **kwargs: Any):
        """ Trains the function encoder on the dataset for some number of epochs.
        
        Args:
        dataset: BaseDataset: The dataset to train on.
        epochs: int: The number of epochs to train for.
        progress_bar: bool: Whether to show a progress bar.
        callback: BaseCallback: A callback to use during training. Can be used to test loss, etc. 
        
        Returns:
        list[float]: The losses at each epoch."""

        # verify dataset is correct
        dataset.check_dataset()
        
        # set device
        device = next(self.function_encoder.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())

        # method to use for representation during training
        assert self.function_encoder.method in ["inner_product", "least_squares"], f"Unknown method: {self.function_encoder.method}"

        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            example_xs, example_ys, query_xs, query_ys, _ = dataset.sample()

            # train average function, if it exists
            if self.function_encoder.average_function is not None:
                # predict averages
                expected_yhats = self.function_encoder.average_function.forward(query_xs).to(device)

                # compute average function loss
                average_function_loss = self.function_encoder._distance(expected_yhats, query_ys, squared=True).mean()
                
                # we only train average function to fit data in general, so block backprop from the basis function loss
                expected_yhats = expected_yhats.detach()
            else:
                expected_yhats = None

            # approximate functions, compute error
            representation, gram = self.function_encoder.compute_representation(example_xs, example_ys, method=self.function_encoder.method, **kwargs)
            y_hats = self.function_encoder.predict(query_xs, representation, precomputed_average_ys=expected_yhats)
            prediction_loss = self.function_encoder._distance(y_hats, query_ys, squared=True).mean()

            # add loss components
            loss = prediction_loss
            if self.function_encoder.method == "least_squares":
                norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1)**2).mean()
                loss = loss + self.function_encoder.regularization_parameter * norm_loss
            if self.function_encoder.average_function is not None:
                loss = loss + average_function_loss
            
            # backprop with gradient clipping
            loss.backward()
            if (epoch + 1) % self.function_encoder.gradient_accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(self.function_encoder.parameters(), 1)
                self.function_encoder.opt.step()
                self.function_encoder.opt.zero_grad()

            # callbacks
            if callback is not None:
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())