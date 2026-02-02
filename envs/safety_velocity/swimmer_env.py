import os
import tempfile
import logging
from typing import Dict, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

DEFAULT_SWIMMER_FRICTION = 0.1
DEFAULT_SWIMMER_TORSO_LENGTH = 0.1
DEFAULT_SWIMMER_MID_LENGTH = 0.1
DEFAULT_SWIMMER_BACK_LENGTH = 0.1
DEFAULT_SWIMMER_HINGE1_GEAR = 150.0
DEFAULT_SWIMMER_HINGE2_GEAR = 150.0

IN_DISTRIBUTION_MULTIPLIER_RANGE = (0.7, 1.3)
OOD_MULTIPLIER_RANGES = ((0.4, 0.7), (1.3, 1.6))

logger = logging.getLogger(__name__)


def _scale_default_range(default_value: float, multiplier_range):
    return default_value * multiplier_range[0], default_value * multiplier_range[1]


def build_default_swimmer_dynamics_variable_ranges(
    is_out_of_distribution: bool,
    in_distribution_multiplier_range=None,
) -> Dict:
    if not is_out_of_distribution:
        multiplier_ranges = [in_distribution_multiplier_range or IN_DISTRIBUTION_MULTIPLIER_RANGE]
    else:
        multiplier_ranges = list(OOD_MULTIPLIER_RANGES)

    def _ranges_for(default_value):
        scaled = tuple(_scale_default_range(default_value, mr) for mr in multiplier_ranges)
        return scaled[0] if len(scaled) == 1 else scaled

    return {
        'friction': _ranges_for(DEFAULT_SWIMMER_FRICTION),
        'torso_length': _ranges_for(DEFAULT_SWIMMER_TORSO_LENGTH),
        'mid_length': _ranges_for(DEFAULT_SWIMMER_MID_LENGTH),
        'back_length': _ranges_for(DEFAULT_SWIMMER_BACK_LENGTH),
        'hinge1_gear': _ranges_for(DEFAULT_SWIMMER_HINGE1_GEAR),
        'hinge2_gear': _ranges_for(DEFAULT_SWIMMER_HINGE2_GEAR),
    }


class VariableSwimmerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'depth_array'], 'render_fps': 30}

    def _extract_is_ood_flag(self, cfg) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return cfg.get('is_out_of_distribution') or cfg.get('IS_OUT_OF_DISTRIBUTION') or cfg.get('env_config', {}).get('is_out_of_distribution', False)
        return bool(getattr(cfg, 'IS_OUT_OF_DISTRIBUTION', getattr(cfg, 'is_out_of_distribution', False)))
    
    def _extract_oracle_flag(self, cfg) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return cfg.get('use_oracle') or cfg.get('USE_ORACLE') or cfg.get('env_config', {}).get('use_oracle', False)
        return bool(getattr(cfg, 'USE_ORACLE', getattr(cfg, 'use_oracle', False)))

    def _extract_multiplier_range(self, cfg):
        if cfg is None:
            return None
        if isinstance(cfg, dict):
            min_mult = cfg.get('MIN_MULT', cfg.get('min_mult'))
            max_mult = cfg.get('MAX_MULT', cfg.get('max_mult'))
            nested = cfg.get('env_config', {})
            if min_mult is None:
                min_mult = nested.get('MIN_MULT', nested.get('min_mult'))
            if max_mult is None:
                max_mult = nested.get('MAX_MULT', nested.get('max_mult'))
        else:
            min_mult = getattr(cfg, 'MIN_MULT', getattr(cfg, 'min_mult', None))
            max_mult = getattr(cfg, 'MAX_MULT', getattr(cfg, 'max_mult', None))
        if min_mult is None or max_mult is None:
            return None
        return (float(min_mult), float(max_mult))

    def __init__(
        self,
        velocity_threshold: Optional[float] = 0.1142,
        *env_args,
        binary_cost: bool = True,
        cost_scale: float = 1.0,
        env_config: Optional[Dict] = None,
        **env_kwargs,
    ):
        super().__init__()
        self.env_args = env_args
        self.env_kwargs = env_kwargs
        nested_env_kwargs = self.env_kwargs.pop('env_kwargs', {})
        self.env_config = env_config if env_config is not None else nested_env_kwargs.get('env_config', None)
        self.is_out_of_distribution = self._extract_is_ood_flag(self.env_config)
        self.oracle = self._extract_oracle_flag(self.env_config)
        self.multiplier_range = self._extract_multiplier_range(self.env_config)
        self.render_mode = env_kwargs.get('render_mode', None)
        self._default_values = {
            'friction': DEFAULT_SWIMMER_FRICTION,
            'torso_length': DEFAULT_SWIMMER_TORSO_LENGTH,
            'mid_length': DEFAULT_SWIMMER_MID_LENGTH,
            'back_length': DEFAULT_SWIMMER_BACK_LENGTH,
            'hinge1_gear': DEFAULT_SWIMMER_HINGE1_GEAR,
            'hinge2_gear': DEFAULT_SWIMMER_HINGE2_GEAR,
        }
        if self.is_out_of_distribution and self.multiplier_range is not None:
            logger.info(
                "OOD enabled: ignoring MIN_MULT/MAX_MULT=%s for OOD ranges %s",
                self.multiplier_range,
                OOD_MULTIPLIER_RANGES,
            )
        self.dynamics_variable_ranges = build_default_swimmer_dynamics_variable_ranges(
            self.is_out_of_distribution,
            in_distribution_multiplier_range=self.multiplier_range,
        )
        self.max_episode_steps = 1000
        self.velocity_threshold = velocity_threshold
        self.binary_cost = binary_cost
        self.cost_scale = cost_scale
        self._last_hidden_parameters_features = None

        if 'friction' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['friction'] = (DEFAULT_SWIMMER_FRICTION, DEFAULT_SWIMMER_FRICTION)
        if 'torso_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['torso_length'] = (DEFAULT_SWIMMER_TORSO_LENGTH, DEFAULT_SWIMMER_TORSO_LENGTH)
        if 'mid_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['mid_length'] = (DEFAULT_SWIMMER_MID_LENGTH, DEFAULT_SWIMMER_MID_LENGTH)
        if 'back_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['back_length'] = (DEFAULT_SWIMMER_BACK_LENGTH, DEFAULT_SWIMMER_BACK_LENGTH)
        if 'hinge1_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['hinge1_gear'] = (DEFAULT_SWIMMER_HINGE1_GEAR, DEFAULT_SWIMMER_HINGE1_GEAR)
        if 'hinge2_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['hinge2_gear'] = (DEFAULT_SWIMMER_HINGE2_GEAR, DEFAULT_SWIMMER_HINGE2_GEAR)

        self.env = gym.make('Swimmer-v5', *self.env_args, **self.env_kwargs)
        _, self._initial_info = self.env.reset()
        self.hidden_parameters_dim = len(self._default_values)
        if not self.oracle:
            n_basis = None
            if isinstance(self.env_config, dict):
                n_basis = self.env_config.get("nbr_of_basis")
            else:
                n_basis = getattr(self.env_config, "NBR_OF_BASIS", None)
            if n_basis is not None:
                self.hidden_parameters_dim = int(n_basis)
        bounds = {'original': (self.env.observation_space.low, self.env.observation_space.high)}   
        bounds['hidden'] = ([-2.] * self.hidden_parameters_dim, [2.] * self.hidden_parameters_dim)

        low = np.concatenate([b[0] for b in bounds.values()])
        high = np.concatenate([b[1] for b in bounds.values()])

        self.observation_space = spaces.Box(low, high)
        self.action_space = self.env.action_space

        base_tmp_dir = tempfile.gettempdir()
        self.xml_path = os.path.join(base_tmp_dir, 'swimmers', str(os.getpid()))
        os.makedirs(self.xml_path, exist_ok=True)
        self._coeffs = np.zeros((1, self.hidden_parameters_dim), dtype=np.float32)

    def set_coefficients(self, coeffs: np.ndarray) -> None:
        if coeffs is None:
            return
        coeffs = np.asarray(coeffs, dtype=np.float32)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        if coeffs.shape[-1] != self.hidden_parameters_dim:
            logger.warning(
                "Ignoring coefficients with dim %s (expected %s)",
                coeffs.shape[-1],
                self.hidden_parameters_dim,
            )
            return
        self._coeffs = coeffs

    def _sample_from_range(self, key: str) -> float:
        value_range = self.dynamics_variable_ranges[key]
        if isinstance(value_range[0], (tuple, list, np.ndarray)):
            low, high = value_range[np.random.randint(len(value_range))]
        else:
            low, high = value_range
        return np.random.uniform(low, high)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.sampled_parameters = {}
        self.sampled_scales = {}

        def _record_sample(key):
            value = self._sample_from_range(key)
            default = self._default_values.get(key, 1.0)
            scale = value / default if default != 0 else None
            self.sampled_parameters[key] = value
            self.sampled_scales[key] = scale
            return value

        friction = _record_sample('friction')
        torso_length = _record_sample('torso_length')
        mid_length = _record_sample('mid_length')
        back_length = _record_sample('back_length')
        hinge1_gear = _record_sample('hinge1_gear')
        hinge2_gear = _record_sample('hinge2_gear')

        path = self.create_xml_file(
            friction,
            torso_length,
            mid_length,
            back_length,
            hinge1_gear,
            hinge2_gear,
        )

        self.env = gym.make('Swimmer-v5', xml_file=path, *self.env_args, **self.env_kwargs)
        obs, info = self.env.reset(seed=seed, options=options)
        obs_dims = {'robot': len(obs)}
        info = dict(info)
        info['obs_dims'] = obs_dims
        self.hidden_parameter_features = np.array([v - 1.0 for v in self.sampled_scales.values()])
        info['hidden_parameters_features'] = self.hidden_parameter_features
        obs = self.augment_state(obs, self.hidden_parameter_features)
        info['hidden_parameters_dim'] = self.hidden_parameters_dim
        info['x_velocity'] = 0.0
        changed = (
            self._last_hidden_parameters_features is not None
            and not np.array_equal(self.hidden_parameter_features, self._last_hidden_parameters_features)
        )
        if not self.is_out_of_distribution:
            expected_range = self.multiplier_range or IN_DISTRIBUTION_MULTIPLIER_RANGE
        else:
            expected_range = OOD_MULTIPLIER_RANGES
        logger.info(
            "Reset: sampled scales min=%s max=%s expected=%s (changed=%s)",
            float(np.min(list(self.sampled_scales.values()))),
            float(np.max(list(self.sampled_scales.values()))),
            expected_range,
            changed,
        )
        self._last_hidden_parameters_features = self.hidden_parameter_features.copy()
        return obs, info
    
    def augment_state(self, obs: np.ndarray, hidden_parameters_features: np.ndarray) -> np.ndarray:
        hidden_info = hidden_parameters_features if self.oracle else self._coeffs.reshape(-1)
        return np.concatenate((obs, hidden_info))

    def _compute_velocity_cost(self, x_velocity: Optional[float]) -> float:
        if self.velocity_threshold is None or x_velocity is None:
            return 0.0
        return float(abs(x_velocity) > self.velocity_threshold) * self.cost_scale

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        x_velocity = info.get('x_velocity')
        if x_velocity is None and hasattr(self.env, 'data'):
            x_velocity = float(self.env.data.qvel[0])
        cost = self._compute_velocity_cost(x_velocity)
        info = dict(info)
        info['velocity_cost'] = cost
        info['velocity_threshold'] = self.velocity_threshold
        info['hidden_parameters_features'] = self.hidden_parameter_features
        info['hidden_parameters_dim'] = self.hidden_parameters_dim
        info['x_velocity'] = x_velocity
        observation = self.augment_state(observation, self.hidden_parameter_features)
        return observation, reward, cost, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def create_xml_file(self, friction, torso_length, mid_length, back_length, hinge1_gear, hinge2_gear):
        file_string = f'''<mujoco model="swimmer">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <size nstack="300000" nuser_geom="1"/>
  <option integrator="RK4" timestep="0.003"/>
  <default>
    <joint armature="0.1" damping="1.0" limited="false"/>
    <geom conaffinity="0" condim="1" friction="{friction} .1 .1" rgba="0.6 0.3 0.5 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 0">
      <geom name="torso" type="capsule" fromto="0 0 0 {torso_length} 0 0" size="0.05"/>
      <joint name="slider1" type="slide" axis="1 0 0"/>
      <joint name="slider2" type="slide" axis="0 1 0"/>
      <joint name="slider3" type="hinge" axis="0 0 1"/>
      <body name="mid" pos="{torso_length} 0 0">
        <geom name="mid" type="capsule" fromto="0 0 0 {mid_length} 0 0" size="0.05"/>
        <joint name="hinge1" type="hinge" axis="0 0 1" pos="0 0 0"/>
        <body name="back" pos="{mid_length} 0 0">
          <geom name="back" type="capsule" fromto="0 0 0 {back_length} 0 0" size="0.05"/>
          <joint name="hinge2" type="hinge" axis="0 0 1" pos="0 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge1" gear="{hinge1_gear}" name="hinge1"/>
    <motor joint="hinge2" gear="{hinge2_gear}" name="hinge2"/>
  </actuator>
</mujoco>'''
        fd, temp_path = tempfile.mkstemp(dir=self.xml_path, suffix='.xml')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(file_string)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            try:
                os.remove(temp_path)
            finally:
                pass
            raise
        return temp_path
