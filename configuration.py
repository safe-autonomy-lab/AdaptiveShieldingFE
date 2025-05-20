from dataclasses import dataclass
from typing import List
import jax.numpy as jnp

@dataclass
class DynamicPredictorConfig:
    MAX_HISTORY: int = 1
    VOLUME: int = 1
    N_BASIS: int = 2
    MAX_LEN: int = 100
    LEARNING_RATE: float = 1e-3
    EPOCH: int = 1000
    ENSEMBLE_SIZE: int = 2
    LEAST_SQUARES: bool = True
    AVERAGE_FUNCTION: bool = True
    HIDDEN_SIZE: int = 512
    USE_ATTENTION: bool = False
    LEARNING_DOMAIN: str = 'ds'

# This will be usedin envs.safety_gymnasium.builder.py, world.py, and etc.
@dataclass
class EnvironmentConfig:
    FIX_HIDDEN_PARAMETERS: bool = False
    USE_ORACLE: bool = False
    USE_FE_REPRESENTATION: bool = False
    NBR_OF_HIDDEN_PARAMETERS: int = 2
    # These are parameters to control the variability of the environment
    MIN_MULT: float = 0.25
    MAX_MULT: float = 1.75
    # for evaluation, this should be negative to sample out-of-distribution parameters
    # MIN_MULT: float = -1.0
    # MAX_MULT: float = -1.0
    # These parameters are used to define the number of gremlins and static obstacles only for Goal2-Tasks
    NBR_OF_GREMLINS: int = 0
    NBR_OF_STATIC_OBSTACLES: int = 0
    NBR_OF_GOALS: int = 1
    ENV_INFO: str = 'SafetyPointGoal2'
    RANGE_LIMIT: float = -1.0 # -1.0 means no range limit
    RADIUS: float = 1.5 # Only used for circle environments
    EXAMPLE_NBR: int = 100
    SPEED_LIMIT: bool = False
    GENERAL_ENV: bool = True
    CIRCLE_ENV: bool = False
    ACTION_DIM: int = 2
