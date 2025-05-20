from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any, List, Optional
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial


class BaseDataset(ABC):
    """Base class for all datasets. Follow this interface to interact with FunctionEncoder.py"""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        output_size: Tuple[int, ...],
        total_n_functions: Union[int, float],
        total_n_samples_per_function: Union[int, float],
        data_type: str,
        n_functions_per_sample: int,
        n_examples_per_sample: int,
        n_points_per_sample: int,
        dtype: jnp.dtype = jnp.float32,
        memory_size: int = 2000, # number of different weights
    ):
        """
        Constructor for BaseDataset

        Args:
            input_size (Tuple[int, ...]): Size of input to the function space, i.e., the number of dimensions of the input
            output_size (Tuple[int, ...]): Size of output of the function space, i.e., the number of dimensions of the output
            total_n_functions (Union[int, float]): Number of functions in this dataset. If functions are sampled from a continuous space, this can be float('inf')
            total_n_samples_per_function (Union[int, float]): Number of data points per function. If data is sampled from a continuous space, this can be float('inf')
            data_type (str): Type of data. Options are "deterministic", "stochastic", or "categorical". Affects which inner product method is used.
            n_functions_per_sample (int): Number of functions per training step. Should be at least 5 or so.
            n_examples_per_sample (int): Number of example points per function per training step. This data is used by the function encoder to compute coefficients.
            n_points_per_sample (int): Number of target points per function per training step. Should be large enough to capture the function's behavior. These points are used to train the function encoder as the target of the prediction, i.e., the MSE.
            dtype (jnp.dtype): Data type for JAX arrays.
        """
        assert len(input_size) >= 1, "input_size must be a tuple of at least one element"
        assert len(output_size) >= 1, "output_size must be a tuple of at least one element"
        assert total_n_functions >= 1, "n_functions must be a positive integer or infinite"
        assert total_n_samples_per_function >= 1, "n_samples_per_function must be a positive integer or infinite"
        assert data_type in ["deterministic", "stochastic", "categorical"]

        self.input_size = input_size
        self.output_size = output_size
        self.n_functions = total_n_functions  # may be infinite
        self.n_samples_per_function = total_n_samples_per_function  # may be infinite
        self.data_type = data_type.lower()
        self.n_functions_per_sample = n_functions_per_sample
        self.n_examples_per_sample = n_examples_per_sample
        self.n_points_per_sample = n_points_per_sample
        self.dtype = dtype

        self.experiences: Dict[Tuple[float, ...], List[Tuple[jnp.ndarray, jnp.ndarray]]] = {}
        self.max_size = memory_size

    @abstractmethod
    def sample(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Sample a batch of functions from the dataset.

        Returns:
            Tuple containing:
            - jnp.ndarray: Example Input data to compute a representation. Shape is (n_functions, n_examples, *input_size)
            - jnp.ndarray: Example Output data to compute a representation. Shape is (n_functions, n_examples, *output_size)
            - jnp.ndarray: Input data to predict outputs for. Shape is (n_functions, n_points, *input_size)
            - jnp.ndarray: Output data, i.e., target of the prediction. Shape is (n_functions, n_points, *output_size)
            - Dict[str, Any]: Additional information about the sampled functions
        """
        pass

    def reset_experiences(self):
        """
        Reset the experiences dictionary.
        """
        self.experiences = {}

    def add_experiences(self, X: Union[jnp.ndarray, List[jnp.ndarray]], 
                        y: Union[jnp.ndarray, List[jnp.ndarray]], 
                        w: Union[jnp.ndarray, List[jnp.ndarray]]) -> None:
        """
        Add new experiences to the dataset. Supports both single and batch additions.

        Args:
            X (Union[jnp.ndarray, List[jnp.ndarray]]): Feature array(s)
            y (Union[jnp.ndarray, List[jnp.ndarray]]): Label array(s)
            w (Union[jnp.ndarray, List[jnp.ndarray]]): Hidden parameter array(s)

        Raises:
            ValueError: If input types or shapes are inconsistent
        """
        if isinstance(X, jnp.ndarray) and isinstance(y, jnp.ndarray):
            X, y = [X], [y]
        elif not (isinstance(X, list) and isinstance(y, list)):
            raise ValueError("Inputs must be either all ndarrays or all lists of ndarrays")

        if not (len(X) == len(y)):
            raise ValueError("Inconsistent number of experiences in input lists")

        if isinstance(w, jnp.ndarray):
            w_key = tuple(float(val) for val in jax.device_get(w.flatten()))
        else:
            w_key = w

        for x, y_i in zip(X, y):
            if x.shape[0] != y_i.shape[0]:
                raise ValueError("Inconsistent number of samples in input arrays")
            self.experiences[w_key] = (x, y_i)

        self._limit_size()

    def _limit_size(self) -> None:
        """
        Limit the dataset size to max_size by removing oldest experiences if necessary.
        This ensures that the dataset doesn't grow beyond the specified maximum size.
        """
        if self.max_size is not None:
            while len(self.experiences) > self.max_size:
                self.experiences.pop(next(iter(self.experiences)))
    
    def get_size(self) -> int:
        """
        Get the current size of the dataset.

        Returns:
            int: Current size of the dataset.
        """
        return len(self.experiences)

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def compute_distances(w1: jnp.ndarray, w2: jnp.ndarray, distance_type: str = 'cosine') -> jnp.ndarray:
        """
        Compute the Euclidean distances between two sets of hidden parameters.
        This method is jit-compiled for improved performance.

        Args:
            w1 (jnp.ndarray): First set of hidden parameters
            w2 (jnp.ndarray): Second set of hidden parameters

        Returns:
            jnp.ndarray: Array of distances
        """
        if distance_type == 'euclidean':
            return jnp.sqrt(jnp.sum((w1 - w2) ** 2, axis=-1))
        elif distance_type == 'cosine':
            # Compute dot product
            dot_product = jnp.sum(w1 * w2, axis=1)
            
            # Compute norms
            norm_w1 = jnp.linalg.norm(w1, axis=1)
            norm_w2 = jnp.linalg.norm(w2)
            
            # Compute cosine similarity
            cosine_similarity = dot_product / (norm_w1 * norm_w2)
            return cosine_similarity
        else:
            raise ValueError(f"Invalid distance type: {distance_type}")
    
    def retrive_similar_hidden_parameters(self, w: jnp.ndarray, epsilon: float) -> List[jnp.ndarray]:
        """
        Retrieve hidden parameters similar to the given w.
        """
        ep_key = jnp.array(list(self.experiences.keys()))
        distances = self.compute_distances(ep_key, w)
        mask = distances < epsilon
        np_mask = np.array(mask)
        selected_keys = ep_key[np_mask]
        return selected_keys
    
    def retrieve_similar_experiences(
        self, w: jnp.ndarray, epsilon: float, batch_size: Optional[int] = None, distance_type: str = 'cosine'
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Retrieve experiences with hidden parameters similar to the given w.
        This method uses vectorized operations for efficient similarity search.

        Args:
            w (jnp.ndarray): Target hidden parameter
            epsilon (float): Maximum distance for similarity
            batch_size (Optional[int]): If provided, process the data in batches of this size

        Returns:
            Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]: Similar X, y, and w arrays

        Raises:
            ValueError: If the dataset is empty or if there's a shape mismatch in the stored data
        """
        similar_X, similar_y, similar_w = [], [], []
        ep = self.experiences
        ep_key = jnp.array(list(self.experiences.keys()))

        if batch_size is None:
            # Process all data at once
            distances = BaseDataset.compute_distances(ep_key, w, distance_type)
            # Calculate the 10th percentile of distances
            percentile = jnp.percentile(distances, 100 * epsilon)
            mask = distances < percentile
            np_mask = np.array(mask)
            selected_keys = ep_key[np_mask]
            for key in selected_keys:
                # Convert key to tuple for dictionary indexing
                w_key = tuple(float(val) for val in jax.device_get(key.flatten()))
                similar_w.append(w_key)
                if w_key in ep:
                    similar_X.append(ep[w_key][0])
                    similar_y.append(ep[w_key][1])
                
        else:
            # Process data in batches
            for i in range(0, len(ep_key), batch_size):
                X_batch = ep[i:i+batch_size][0]
                y_batch = ep[i:i+batch_size][1]    
                w_batch = ep_key[i:i+batch_size]
                
                distances = self.compute_distances(w_batch, w, distance_type)
                mask = distances < epsilon
                mask = np.array(mask)
                
                if jnp.any(mask):
                    similar_X.append(X_batch[mask])
                    similar_y.append(y_batch[mask])
                    similar_w.append(w_batch[mask])

        if len(similar_X) > 0:
            similar_X = jnp.concatenate(similar_X, axis=0)
            similar_y = jnp.concatenate(similar_y, axis=0)

        return similar_X, similar_y, similar_w

    def check_dataset(self):
        """
        Verify that a dataset is correctly implemented. Throws error if violated.
        It is advised against overriding this method, as it is used to verify that the dataset is implemented correctly.
        However, if your use case is very different, you may need to.
        """
        out = self.sample()
        assert len(out) == 5, f"Expected 5 outputs, got {len(out)}"
        
        example_xs, example_ys, xs, ys, info = out
        assert isinstance(example_xs, jnp.ndarray), f"Expected example_xs to be a jnp.ndarray, got {type(example_xs)}"
        assert isinstance(example_ys, jnp.ndarray), f"Expected example_ys to be a jnp.ndarray, got {type(example_ys)}"
        assert isinstance(xs, jnp.ndarray), f"Expected xs to be a jnp.ndarray, got {type(xs)}"
        assert isinstance(ys, jnp.ndarray), f"Expected ys to be a jnp.ndarray, got {type(ys)}"
        # the second case is for the case when the hidden parameters are added to the input
        assert example_xs.shape in [(self.n_functions_per_sample, self.n_examples_per_sample, *self.input_size), (self.n_functions_per_sample, self.n_examples_per_sample, self.input_size[0] + 1)], f"Expected example_xs shape to be {(self.n_functions_per_sample, self.n_examples_per_sample, *self.input_size)}, got {example_xs.shape}"
        assert example_ys.shape == (self.n_functions_per_sample, self.n_examples_per_sample, *self.output_size), f"Expected example_ys shape to be {(self.n_functions_per_sample, self.n_examples_per_sample, *self.output_size)}, got {example_ys.shape}"
        assert xs.shape in [(self.n_functions_per_sample, self.n_points_per_sample, *self.input_size), (self.n_functions_per_sample, self.n_points_per_sample, self.input_size[0] + 1)], f"Expected xs shape to be {(self.n_functions_per_sample, self.n_points_per_sample, *self.input_size)}, got {xs.shape}"
        assert ys.shape == (self.n_functions_per_sample, self.n_points_per_sample, *self.output_size), f"Expected ys shape to be {(self.n_functions_per_sample, self.n_points_per_sample, *self.output_size)}, got {ys.shape}"
        assert isinstance(info, dict), f"Expected info to be a dict, got {type(info)}"
        assert example_xs.dtype == example_ys.dtype == xs.dtype == ys.dtype == self.dtype, f"Expected all arrays to have dtype {self.dtype}, got {example_xs.dtype}, {example_ys.dtype}, {xs.dtype}, {ys.dtype}"


class TimeseriesDataset(BaseDataset):
    """
    A dataset class for generating timeseries data using JAX.
    """
    def __init__(
        self,
        stacked_history: jnp.ndarray,
        future_positions: jnp.ndarray,
        eval_stacked_history: jnp.ndarray,
        eval_future_positions: jnp.ndarray,
        n_examples_per_sample: int = 1,
        dtype: jnp.dtype = jnp.float32,
    ): 
        # assume all episodes have the same number of functions (same number of pedestrians)
        n_functions_per_sample = stacked_history[0].shape[0]
        n_points_per_sample = stacked_history[0].shape[1] - n_examples_per_sample
        input_size = stacked_history[0].shape[-1]
        output_size = future_positions[0].shape[-1]
        self.total_train_episodes = len(stacked_history)
        super().__init__(
            input_size=(input_size, ),
            output_size=(output_size, ),
            total_n_functions=float("inf"),
            total_n_samples_per_function=float("inf"),
            data_type="deterministic",
            n_functions_per_sample=n_functions_per_sample,
            n_examples_per_sample=n_examples_per_sample,
            n_points_per_sample=n_points_per_sample,
            dtype=dtype,
        )
        self.stacked_history = stacked_history
        self.future_positions = future_positions
        self.eval_stacked_history = eval_stacked_history
        self.eval_future_positions = eval_future_positions
        self.total_eval_episodes = len(eval_stacked_history)

    def sample(self, mode: str = "train", episode_index: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Generate a sample of transitions.

        Returns:
            A tuple containing:
            - states: Input values for example points.
            - actions: Output values for example points.
            - next_states: Input values for evaluation points.
            - params: Dictionary of function parameters (As, Bs, Cs).
        """
        assert mode in ["train", "eval"], f"Invalid mode: {mode}. Please specify 'train' or 'eval'."
        
        if mode == "train":
            assert episode_index < self.total_train_episodes, f"Episode index {episode_index} is out of bounds for training dataset with {self.total_train_episodes} episodes."
            example_xs = self.stacked_history[episode_index][:, :, :]
            example_ys = self.future_positions[episode_index][:, :, :]
            xs = self.stacked_history[episode_index][:, :, :]
            ys = self.future_positions[episode_index][:, :, :]            
            
        elif mode == "eval":
            assert episode_index < self.total_eval_episodes, f"Episode index {episode_index} is out of bounds for evaluation dataset with {self.total_eval_episodes} episodes."
            example_xs = self.eval_stacked_history[episode_index][:, :, :]
            example_ys = self.eval_future_positions[episode_index][:, :, :]
            xs = self.eval_stacked_history[episode_index][:, :, :]
            ys = self.eval_future_positions[episode_index][:, :, :]

        return example_xs, example_ys, xs, ys, {}


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
        dimension_cut: int = 0,
        dtype: jnp.dtype = jnp.float32,
    ): 
        
        n_functions_per_sample = transition_states_and_actions.shape[0]
        n_points_per_sample = transition_states_and_actions.shape[1] - n_examples_per_sample
        input_size = transition_states_and_actions.shape[-1] - dimension_cut
        output_size = transition_next_states.shape[-1]
        self.dimension_cut = dimension_cut
        super().__init__(
            input_size=(input_size, ),
            output_size=(output_size, ),
            total_n_functions=float("inf"),
            total_n_samples_per_function=float("inf"),
            data_type="deterministic",
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
            - params: Dictionary of function parameters (As, Bs, Cs).
        """
        assert mode in ["train", "eval"], f"Invalid mode: {mode}. Please specify 'train' or 'eval'."
        
        if mode == "train":
            example_xs = self.transition_states_and_actions[:, :self.n_examples_per_sample, self.dimension_cut:]
            example_ys = self.transition_next_states[:, :self.n_examples_per_sample, :]
            xs = self.transition_states_and_actions[:, self.n_examples_per_sample:, self.dimension_cut:]
            ys = self.transition_next_states[:, self.n_examples_per_sample:, :]            
            hiddens = self.hiddens
            if add_hidden_params:
                hiddens = jnp.array(hiddens).reshape(-1, 1, self.n_hiddens)
                hidden_repeat = jnp.repeat(hiddens, xs.shape[1], axis=1)
                hidden_repeat_for_example = jnp.repeat(hiddens, example_xs.shape[1], axis=1)
                example_xs = jnp.concatenate([example_xs, hidden_repeat_for_example], axis=-1)
                xs = jnp.concatenate([xs, hidden_repeat], axis=-1)   

        elif mode == "eval":
            example_xs = self.eval_transition_states_and_actions[:, :self.n_examples_per_sample, self.dimension_cut:]
            example_ys = self.eval_transition_next_states[:, :self.n_examples_per_sample, :]
            xs = self.eval_transition_states_and_actions[:, self.n_examples_per_sample:, self.dimension_cut:]
            ys = self.eval_transition_next_states[:, self.n_examples_per_sample:, :]
            hiddens = self.eval_hiddens
            if add_hidden_params:
                hiddens = jnp.array(hiddens).reshape(-1, 1, self.n_hiddens)
                hidden_repeat = jnp.repeat(hiddens, xs.shape[1], axis=1)
                hidden_repeat_for_example = jnp.repeat(hiddens, example_xs.shape[1], axis=1)
                example_xs = jnp.concatenate([example_xs, hidden_repeat_for_example], axis=-1)
                xs = jnp.concatenate([xs, hidden_repeat], axis=-1)   

        return example_xs, example_ys, xs, ys, {"hiddens": hiddens}
    

class ExperienceDataset:
    def __init__(self, input_size: int, output_size: int, hidden_param_size: int, max_size: Optional[int] = None):
        """
        Initialize an empty ExperienceDataset.

        Args:
            max_size (Optional[int]): Maximum number of experiences to store. If None, no limit is applied.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_param_size = hidden_param_size
        self.max_size = max_size
    
    def set_offline_data(self, X: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray) -> None:
        self.X = X
        self.y = y
        self.w = w
        assert len(self.X) == len(self.y) == len(self.w)
        assert self.X.shape[-1] == self.input_size
        assert self.y.shape[-1] == self.output_size
        assert self.w.shape[-1] == self.hidden_param_size

    def add_experiences(self, X: Union[jnp.ndarray, List[jnp.ndarray]], 
                        y: Union[jnp.ndarray, List[jnp.ndarray]], 
                        w: Union[jnp.ndarray, List[jnp.ndarray]]) -> None:
        """
        Add new experiences to the dataset. Supports both single and batch additions.

        Args:
            X (Union[jnp.ndarray, List[jnp.ndarray]]): Feature array(s)
            y (Union[jnp.ndarray, List[jnp.ndarray]]): Label array(s)
            w (Union[jnp.ndarray, List[jnp.ndarray]]): Hidden parameter array(s)

        Raises:
            ValueError: If input types or shapes are inconsistent
        """
        assert X.shape[-1] == self.input_size, f"X.shape[-1] = {X.shape[-1]}, self.input_size = {self.input_size}"
        assert y.shape[-1] == self.output_size, f"y.shape[-1] = {y.shape[-1]}, self.output_size = {self.output_size}"   
        assert w.shape[-1] == self.hidden_param_size, f"w.shape[-1] = {w.shape[-1]}, self.hidden_param_size = {self.hidden_param_size}"

        if isinstance(X, jnp.ndarray) and isinstance(y, jnp.ndarray) and isinstance(w, jnp.ndarray):
            X, y, w = [X], [y], [w]
        elif not (isinstance(X, list) and isinstance(y, list) and isinstance(w, list)):
            raise ValueError("Inputs must be either all ndarrays or all lists of ndarrays")

        if not (len(X) == len(y) == len(w)):
            raise ValueError("Inconsistent number of experiences in input lists")

        for x, y_i, w_i in zip(X, y, w):
            if x.shape[0] != y_i.shape[0] or x.shape[0] != w_i.shape[0]:
                raise ValueError("Inconsistent number of samples in input arrays")

            self.X = jnp.concatenate([self.X, x], axis=0)
            self.y = jnp.concatenate([self.y, y_i], axis=0)
            self.w = jnp.concatenate([self.w, w_i], axis=0)

        self._limit_size()

    def _limit_size(self) -> None:
        """
        Limit the dataset size to max_size by removing oldest experiences if necessary.
        This ensures that the dataset doesn't grow beyond the specified maximum size.
        """
        if self.max_size is not None:
            while len(self.X) > self.max_size:
                self.X.pop(0)
                self.y.pop(0)
                self.w.pop(0)

    @staticmethod
    @jax.jit
    def compute_distances(w1: jnp.ndarray, w2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Euclidean distances between two sets of hidden parameters.
        This method is jit-compiled for improved performance.

        Args:
            w1 (jnp.ndarray): First set of hidden parameters
            w2 (jnp.ndarray): Second set of hidden parameters

        Returns:
            jnp.ndarray: Array of distances
        """
        return jnp.sqrt(jnp.sum((w1 - w2) ** 2, axis=-1))

    def get_size(self) -> int:
        """
        Get the current number of experiences in the dataset.

        Returns:
            int: Number of experiences in the dataset
        """
        return len(self.X)

    def retrieve_similar_experiences(
        self, w: jnp.ndarray, epsilon: float, batch_size: Optional[int] = None
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Retrieve experiences with hidden parameters similar to the given w.
        This method uses vectorized operations for efficient similarity search.

        Args:
            w (jnp.ndarray): Target hidden parameter
            epsilon (float): Maximum distance for similarity
            batch_size (Optional[int]): If provided, process the data in batches of this size

        Returns:
            Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]: Similar X, y, and w arrays

        Raises:
            ValueError: If the dataset is empty or if there's a shape mismatch in the stored data
        """
        
        similar_X, similar_y, similar_w = [], [], []

        if batch_size is None:
            # Process all data at once
            distances = self.compute_distances(self.w, w)
            mask = distances < epsilon
            similar_X.append(self.X[mask])
            similar_y.append(self.y[mask])
            similar_w.append(self.w[mask])
        else:
            # Process data in batches
            for i in range(0, len(self.X), batch_size):
                X_batch = self.X[i:i+batch_size]
                y_batch = self.y[i:i+batch_size]
                w_batch = self.w[i:i+batch_size]
                
                distances = self.compute_distances(w_batch, w)
                mask = distances < epsilon
                
                if jnp.any(mask):
                    similar_X.append(X_batch[mask])
                    similar_y.append(y_batch[mask])
                    similar_w.append(w_batch[mask])

        similar_X = jnp.array(similar_X) # dtype=object for variable shapes
        similar_y = jnp.array(similar_y)
        similar_w = jnp.array(similar_w)

        key = jax.random.PRNGKey(0)  # Create a PRNG key (use a different seed if needed)
        # Shuffle the indices
        indices = jnp.arange(len(similar_X))
        indices = jax.random.permutation(key, indices)

        # Shuffle the arrays using the shuffled indices
        similar_X = similar_X[indices]
        similar_y = similar_y[indices]
        similar_w = similar_w[indices]

        return similar_X, similar_y, similar_w # Return shuffled NumPy arrays (or 
