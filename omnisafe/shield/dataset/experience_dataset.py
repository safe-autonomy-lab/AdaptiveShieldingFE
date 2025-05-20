import jax.numpy as jnp
from typing import Optional, List, Union


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

    def get_size(self) -> int:
        """
        Get the current number of experiences in the dataset.

        Returns:
            int: Number of experiences in the dataset
        """
        return len(self.X)