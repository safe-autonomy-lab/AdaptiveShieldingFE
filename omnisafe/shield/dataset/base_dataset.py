from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any, List
import jax.numpy as jnp
import jax

class BaseDataset(ABC):
    """Base class for all datasets. Follow this interface to interact with FunctionEncoder.py"""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        output_size: Tuple[int, ...],
        total_n_functions: Union[int, float],
        total_n_samples_per_function: Union[int, float],
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

        self.input_size = input_size
        self.output_size = output_size
        self.n_functions = total_n_functions  # may be infinite
        self.n_samples_per_function = total_n_samples_per_function  # may be infinite
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