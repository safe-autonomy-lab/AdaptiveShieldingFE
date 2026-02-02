from dataclasses import dataclass

# This will be usedin envs.safety_gymnasium.builder.py, world.py, and etc.
@dataclass
class EnvironmentConfig:
    FIX_HIDDEN_PARAMETERS: bool = False
    IS_OUT_OF_DISTRIBUTION: bool = False
    USE_ORACLE: bool = False    
    # These are parameters to control the variability of the environment
    MIN_MULT: float = 0.3
    MAX_MULT: float = 1.7
    ENV_ID: str = 'SafetyPointGoal1-v0'
    # These parameters are used to define the number of gremlins and static obstacles only for Goal2-Tasks
    NBR_OF_GREMLINS: int = 1
    NBR_OF_GOALS: int = 1
    NBR_OF_HAZARDS: int = 1
    NBR_OF_PILLARS: int = 1
    NBR_OF_VASES: int = 1
    PLACEMENT_EXTENTS: int = 2
    NBR_OF_BASIS: int = 2

