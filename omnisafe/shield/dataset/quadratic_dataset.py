from typing import Tuple, Dict
import jax.numpy as jnp
from jax import random

from .base_dataset import BaseDataset


class QuadraticDataset(BaseDataset):
    """
    A dataset class for generating quadratic function data using JAX.
    """

    def __init__(
        self,
        a_range: Tuple[float, float] = (-3, 3),
        b_range: Tuple[float, float] = (-3, 3),
        c_range: Tuple[float, float] = (-3, 3),
        input_range: Tuple[float, float] = (-10, 10),
        n_functions_per_sample: int = 10,
        n_examples_per_sample: int = 100,
        n_points_per_sample: int = 10_000,
        dtype: jnp.dtype = jnp.float32,
    ):
        """
        Initialize the QuadraticDataset.

        Args:
            a_range: Range for the 'a' coefficient in ax^2 + bx + c.
            b_range: Range for the 'b' coefficient in ax^2 + bx + c.
            c_range: Range for the 'c' coefficient in ax^2 + bx + c.
            input_range: Range for the input values.
            n_functions_per_sample: Number of functions to generate per sample.
            n_examples_per_sample: Number of example points per function.
            n_points_per_sample: Number of evaluation points per function.
            dtype: Data type for JAX arrays.
        """
        super().__init__(
            input_size=(1,),
            output_size=(1,),
            total_n_functions=float("inf"),
            total_n_samples_per_function=float("inf"),
            n_functions_per_sample=n_functions_per_sample,
            n_examples_per_sample=n_examples_per_sample,
            n_points_per_sample=n_points_per_sample,
            dtype=dtype,
        )
        self.a_range = jnp.array(a_range, dtype=self.dtype)
        self.b_range = jnp.array(b_range, dtype=self.dtype)
        self.c_range = jnp.array(c_range, dtype=self.dtype)
        self.input_range = jnp.array(input_range, dtype=self.dtype)

    def sample(self, mode='train') -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Generate a sample of quadratic functions and their evaluations.

        Returns:
            A tuple containing:
            - example_xs: Input values for example points.
            - example_ys: Output values for example points.
            - xs: Input values for evaluation points.
            - ys: Output values for evaluation points.
            - params: Dictionary of function parameters (As, Bs, Cs).
        """
        key = random.PRNGKey(0)  # You might want to pass this as an argument or use a different seed mechanism

        n_functions = self.n_functions_per_sample
        n_examples = self.n_examples_per_sample
        n_points = self.n_points_per_sample

        # Generate n_functions sets of coefficients
        key, subkey = random.split(key)
        As = random.uniform(subkey, (n_functions, 1), minval=self.a_range[0], maxval=self.a_range[1], dtype=self.dtype)
        key, subkey = random.split(key)
        Bs = random.uniform(subkey, (n_functions, 1), minval=self.b_range[0], maxval=self.b_range[1], dtype=self.dtype)
        key, subkey = random.split(key)
        Cs = random.uniform(subkey, (n_functions, 1), minval=self.c_range[0], maxval=self.c_range[1], dtype=self.dtype)

        # Generate n_samples_per_function samples for each function
        key, subkey = random.split(key)
        xs = random.uniform(subkey, (n_functions, n_points, *self.input_size), minval=self.input_range[0], maxval=self.input_range[1], dtype=self.dtype)
        key, subkey = random.split(key)
        example_xs = random.uniform(subkey, (n_functions, n_examples, *self.input_size), minval=self.input_range[0], maxval=self.input_range[1], dtype=self.dtype)

        # Compute the corresponding ys
        ys = As[:, jnp.newaxis] * xs ** 2 + Bs[:, jnp.newaxis] * xs + Cs[:, jnp.newaxis]
        example_ys = As[:, jnp.newaxis] * example_xs ** 2 + Bs[:, jnp.newaxis] * example_xs + Cs[:, jnp.newaxis]

        return example_xs, example_ys, xs, ys, {"As": As, "Bs": Bs, "Cs": Cs}