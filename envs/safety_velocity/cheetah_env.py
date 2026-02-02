import os
import tempfile
import logging
from typing import Dict, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.safety_gymnasium.configuration import EnvironmentConfig
# default values specified as constants
DEFAULT_FRICTION = 0.4
DEFAULT_TORSO_LENGTH = 1
DEFAULT_BTHIGH_LENGTH = .145
DEFAULT_BSHIN_LENGTH = .15
DEFAULT_BFOOT_LENGTH = .094
DEFAULT_FTHIGH_LENGTH = .133
DEFAULT_FSHIN_LENGTH = .106
DEFAULT_FFOOT_LENGTH = .07
DEFAULT_BTHIGH_GEAR = 120
DEFAULT_BSHIN_GEAR = 90
DEFAULT_BFOOT_GEAR = 60
DEFAULT_FTHIGH_GEAR = 120
DEFAULT_FSHIN_GEAR = 60
DEFAULT_FFOOT_GEAR = 30

# default multiplier splits
IN_DISTRIBUTION_MULTIPLIER_RANGE = (0.7, 1.3)
OOD_MULTIPLIER_RANGES = ((0.4, 0.7), (1.3, 1.6))

logger = logging.getLogger(__name__)


def _scale_default_range(default_value: float, multiplier_range):
    return default_value * multiplier_range[0], default_value * multiplier_range[1]


def build_default_dynamics_variable_ranges(
    is_out_of_distribution: bool,
    in_distribution_multiplier_range=None,
) -> Dict:
    '''
    Helper to build dynamics ranges from default MuJoCo parameters.

    :param distribution: Either "in_distribution" or "ood". In-distribution multiplies
        defaults by 0.7-1.3. OOD uses two disjoint ranges: 0.4-0.7 and 1.3-1.6.
    '''
    if not is_out_of_distribution:
        multiplier_ranges = [in_distribution_multiplier_range or IN_DISTRIBUTION_MULTIPLIER_RANGE]
    else:
        multiplier_ranges = list(OOD_MULTIPLIER_RANGES)
    
    def _ranges_for(default_value):
        scaled = tuple(_scale_default_range(default_value, mr) for mr in multiplier_ranges)
        return scaled[0] if len(scaled) == 1 else scaled

    return {
        'friction': _ranges_for(DEFAULT_FRICTION),
        'torso_length': _ranges_for(DEFAULT_TORSO_LENGTH),
        'bthigh_length': _ranges_for(DEFAULT_BTHIGH_LENGTH),
        'bshin_length': _ranges_for(DEFAULT_BSHIN_LENGTH),
        'bfoot_length': _ranges_for(DEFAULT_BFOOT_LENGTH),
        'fthigh_length': _ranges_for(DEFAULT_FTHIGH_LENGTH),
        'fshin_length': _ranges_for(DEFAULT_FSHIN_LENGTH),
        'ffoot_length': _ranges_for(DEFAULT_FFOOT_LENGTH),
        'bthigh_gear': _ranges_for(DEFAULT_BTHIGH_GEAR),
        'bshin_gear': _ranges_for(DEFAULT_BSHIN_GEAR),
        'bfoot_gear': _ranges_for(DEFAULT_BFOOT_GEAR),
        'fthigh_gear': _ranges_for(DEFAULT_FTHIGH_GEAR),
        'fshin_gear': _ranges_for(DEFAULT_FSHIN_GEAR),
        'ffoot_gear': _ranges_for(DEFAULT_FFOOT_GEAR),
    }


class VariableCheetahEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'depth_array'], 'render_fps': 30}

    def _extract_is_ood_flag(self, cfg) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return cfg.get('is_out_of_distribution')
        return bool(cfg.IS_OUT_OF_DISTRIBUTION)
    
    def _extract_oracle_flag(self, cfg) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return cfg.get('use_oracle')
        return bool(cfg.USE_ORACLE)

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
        velocity_threshold: Optional[float] = 2.0,
        *env_args,
        binary_cost: bool = True,
        cost_scale: float = 1.0,
        env_config: Optional[Dict] = None,
        **env_kwargs,
    ):
        '''

        :param distribution: If dynamics_variable_ranges is not provided, builds ranges using
        "in_distribution" (0.7-1.3 multipliers) or "ood" (0.4-0.7 and 1.3-1.6 multipliers).
        '''
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
            'friction': DEFAULT_FRICTION,
            'torso_length': DEFAULT_TORSO_LENGTH,
            'bthigh_length': DEFAULT_BTHIGH_LENGTH,
            'bshin_length': DEFAULT_BSHIN_LENGTH,
            'bfoot_length': DEFAULT_BFOOT_LENGTH,
            'fthigh_length': DEFAULT_FTHIGH_LENGTH,
            'fshin_length': DEFAULT_FSHIN_LENGTH,
            'ffoot_length': DEFAULT_FFOOT_LENGTH,
            'bthigh_gear': DEFAULT_BTHIGH_GEAR,
            'bshin_gear': DEFAULT_BSHIN_GEAR,
            'bfoot_gear': DEFAULT_BFOOT_GEAR,
            'fthigh_gear': DEFAULT_FTHIGH_GEAR,
            'fshin_gear': DEFAULT_FSHIN_GEAR,
            'ffoot_gear': DEFAULT_FFOOT_GEAR,
        }
        self.hidden_parameters_dim = len(self._default_values)
        if self.is_out_of_distribution and self.multiplier_range is not None:
            logger.info(
                "OOD enabled: ignoring MIN_MULT/MAX_MULT=%s for OOD ranges %s",
                self.multiplier_range,
                OOD_MULTIPLIER_RANGES,
            )
        self.dynamics_variable_ranges = build_default_dynamics_variable_ranges(
            self.is_out_of_distribution,
            in_distribution_multiplier_range=self.multiplier_range,
        )
        self.max_episode_steps = 1000
        self.velocity_threshold = velocity_threshold
        self.binary_cost = binary_cost
        self.cost_scale = cost_scale
        self._last_hidden_parameters_features = None

        # append defaults if not specified
        if 'friction' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['friction'] = (DEFAULT_FRICTION, DEFAULT_FRICTION)
        if 'torso_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['torso_length'] = (DEFAULT_TORSO_LENGTH, DEFAULT_TORSO_LENGTH)
        if 'bthigh_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bthigh_length'] = (DEFAULT_BTHIGH_LENGTH, DEFAULT_BTHIGH_LENGTH)
        if 'bshin_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bshin_length'] = (DEFAULT_BSHIN_LENGTH, DEFAULT_BSHIN_LENGTH)
        if 'bfoot_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bfoot_length'] = (DEFAULT_BFOOT_LENGTH, DEFAULT_BFOOT_LENGTH)
        if 'fthigh_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['fthigh_length'] = (DEFAULT_FTHIGH_LENGTH, DEFAULT_FTHIGH_LENGTH)
        if 'fshin_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['fshin_length'] = (DEFAULT_FSHIN_LENGTH, DEFAULT_FSHIN_LENGTH)
        if 'ffoot_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['ffoot_length'] = (DEFAULT_FFOOT_LENGTH, DEFAULT_FFOOT_LENGTH)
        if 'bthigh_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bthigh_gear'] = (DEFAULT_BTHIGH_GEAR, DEFAULT_BTHIGH_GEAR)
        if 'bshin_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bshin_gear'] = (DEFAULT_BSHIN_GEAR, DEFAULT_BSHIN_GEAR)
        if 'bfoot_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bfoot_gear'] = (DEFAULT_BFOOT_GEAR, DEFAULT_BFOOT_GEAR)
        if 'fthigh_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['fthigh_gear'] = (DEFAULT_FTHIGH_GEAR, DEFAULT_FTHIGH_GEAR)
        if 'fshin_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['fshin_gear'] = (DEFAULT_FSHIN_GEAR, DEFAULT_FSHIN_GEAR)
        if 'ffoot_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['ffoot_gear'] = (DEFAULT_FFOOT_GEAR, DEFAULT_FFOOT_GEAR)

        # placeholder variable
        self.env =  gym.make('HalfCheetah-v5', *self.env_args, **self.env_kwargs)
        _, self._initial_info = self.env.reset()
        
        if not self.oracle:
            n_basis = None
            if isinstance(self.env_config, EnvironmentConfig):
                n_basis = self.env_config.NBR_OF_BASIS
            elif isinstance(self.env_config, dict):
                n_basis = self.env_config.get("nbr_of_basis")
            else:
                n_basis = getattr(self.env_config, "NBR_OF_BASIS", None)
            if n_basis is not None:
                self.hidden_parameters_dim = int(n_basis)

        self.action_space = self.env.action_space
        bounds = {'original': (self.env.observation_space.low, self.env.observation_space.high)}   
        bounds['hidden'] = ([-2.] * self.hidden_parameters_dim, [2.] * self.hidden_parameters_dim)

        low = np.concatenate([b[0] for b in bounds.values()])
        high = np.concatenate([b[1] for b in bounds.values()])

        self.observation_space = spaces.Box(low, high)
        # path to write xml to
        base_tmp_dir = tempfile.gettempdir()
        # isolate per-process to avoid clashes when vectorized envs fork workers
        self.xml_path = os.path.join(base_tmp_dir, 'half_cheetahs', str(os.getpid()))
        os.makedirs(self.xml_path, exist_ok=True)
        self._coeffs = np.zeros((1, self.hidden_parameters_dim), dtype=np.float32)

    def set_coefficients(self, coeffs: np.ndarray) -> None:
        if coeffs is None:
            return
        coeffs = np.asarray(coeffs, dtype=np.float32)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        self._coeffs = coeffs

    def _sample_from_range(self, key: str) -> float:
        value_range = self.dynamics_variable_ranges[key]
        if isinstance(value_range[0], (tuple, list, np.ndarray)):
            low, high = value_range[np.random.randint(len(value_range))]
        else:
            low, high = value_range
        return np.random.uniform(low, high)

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # sample env parameters
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
        bthigh_length = _record_sample('bthigh_length')
        bshin_length = _record_sample('bshin_length')
        bfoot_length = _record_sample('bfoot_length')
        fthigh_length = _record_sample('fthigh_length')
        fshin_length = _record_sample('fshin_length')
        ffoot_length = _record_sample('ffoot_length')
        bthigh_gear = _record_sample('bthigh_gear')
        bshin_gear = _record_sample('bshin_gear')
        bfoot_gear = _record_sample('bfoot_gear')
        fthigh_gear = _record_sample('fthigh_gear')
        fshin_gear = _record_sample('fshin_gear')
        ffoot_gear = _record_sample('ffoot_gear')

        # create xml file for these parameters
        path = self.create_xml_file(friction, torso_length, bthigh_length, bshin_length, bfoot_length, fthigh_length, fshin_length, ffoot_length, bthigh_gear, bshin_gear, bfoot_gear, fthigh_gear, fshin_gear, ffoot_gear)

        # load env with this xml file
        self.env = gym.make('HalfCheetah-v5', xml_file=path, *self.env_args, **self.env_kwargs)
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info['obs_dims'] = {'robot': len(obs)}
        info['robot'] = slice(0, len(obs))
        info['hidden_param_slices'] = (len(obs), len(obs) + self.hidden_parameters_dim)
        # normalize the feature, mean shift
        self.hidden_parameter_features = np.array([v - 1.0 for v in self.sampled_scales.values()])
        info['hidden_parameters_features'] = self.hidden_parameter_features
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
        obs = self.augment_state(obs, self.hidden_parameter_features)
        return obs, info

    def augment_state(self, obs: np.ndarray, hidden_parameters_features: np.ndarray) -> np.ndarray:
        hidden_info = hidden_parameters_features if self.oracle else self._coeffs.reshape(-1)
        return np.concatenate((obs, hidden_info))

    def _compute_velocity_cost(self, x_velocity: Optional[float]) -> float:
        if self.velocity_threshold is None or x_velocity is None:
            return 0.0
        # cost is positive if velocity is greater than threshold, negative if less than threshold
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

    # generates a custom file for these parameters and writes it to tmp. Returns a path.
    # Note that the constants in this file are the defaults, which  I have rescaled based on the lengths specified.
    def create_xml_file(self, friction, torso_length, bthigh_length, bshin_length, bfoot_length, fthigh_length, fshin_length, ffoot_length, bthigh_gear, bshin_gear, bfoot_gear, fthigh_gear, fshin_gear, ffoot_gear):
        file_string = f'''<!-- Cheetah Model

    The state space is populated with joints in the order that they are
    defined in this file. The actuators also operate on joints.

    State-Space (name/joint/parameter):
        - rootx     slider      position (m)
        - rootz     slider      position (m)
        - rooty     hinge       angle (rad)
        - bthigh    hinge       angle (rad)
        - bshin     hinge       angle (rad)
        - bfoot     hinge       angle (rad)
        - fthigh    hinge       angle (rad)
        - fshin     hinge       angle (rad)
        - ffoot     hinge       angle (rad)
        - rootx     slider      velocity (m/s)
        - rootz     slider      velocity (m/s)
        - rooty     hinge       angular velocity (rad/s)
        - bthigh    hinge       angular velocity (rad/s)
        - bshin     hinge       angular velocity (rad/s)
        - bfoot     hinge       angular velocity (rad/s)
        - fthigh    hinge       angular velocity (rad/s)
        - fshin     hinge       angular velocity (rad/s)
        - ffoot     hinge       angular velocity (rad/s)

    Actuators (name/actuator/parameter):
        - bthigh    hinge       torque (N m)
        - bshin     hinge       torque (N m)
        - bfoot     hinge       torque (N m)
        - fthigh    hinge       torque (N m)
        - fshin     hinge       torque (N m)
        - ffoot     hinge       torque (N m)

-->
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction="{friction} .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="{-0.5 * torso_length/DEFAULT_TORSO_LENGTH} 0 0 {0.5 * torso_length/DEFAULT_TORSO_LENGTH} 0 0" name="torso" size="0.046" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos="{0.5 * torso_length/DEFAULT_TORSO_LENGTH + 0.1} 0 .1" size="0.046 0.15" type="capsule"/>
      <!-- <site name='tip'  pos='.15 0 .11'/>-->
      <body name="bthigh" pos="{-0.5 * torso_length/DEFAULT_TORSO_LENGTH} 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos="{.1 * (bthigh_length/DEFAULT_BTHIGH_LENGTH)} 0 {-.13* (bthigh_length/DEFAULT_BTHIGH_LENGTH)}" size="0.046 {bthigh_length}" type="capsule"/>
        <body name="bshin" pos="{.16 * (bthigh_length/DEFAULT_BTHIGH_LENGTH)} 0 {-.25  * (bthigh_length/DEFAULT_BTHIGH_LENGTH)}">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="{-.14 * bshin_length/DEFAULT_BSHIN_LENGTH} 0 {-.07*bshin_length/DEFAULT_BSHIN_LENGTH}" rgba="0.9 0.6 0.6 1" size="0.046 {bshin_length}" type="capsule"/>
          <body name="bfoot" pos="{-.28 * bshin_length/DEFAULT_BSHIN_LENGTH} 0 {-.14 * bshin_length/DEFAULT_BSHIN_LENGTH}">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos="{.03 * bfoot_length / DEFAULT_BFOOT_LENGTH} 0 {-.097 * bfoot_length / DEFAULT_BFOOT_LENGTH}" rgba="0.9 0.6 0.6 1" size="0.046 {bfoot_length}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos="{0.5 * torso_length/DEFAULT_TORSO_LENGTH} 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="{-.07 * fthigh_length/DEFAULT_FTHIGH_LENGTH} 0 {-.12 * fthigh_length/DEFAULT_FTHIGH_LENGTH}" size="0.046 {fthigh_length}" type="capsule"/>
        <body name="fshin" pos="{-.14 * fthigh_length/DEFAULT_FTHIGH_LENGTH} 0 {-.24 * fthigh_length/DEFAULT_FTHIGH_LENGTH}">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos="{.065 * fshin_length/DEFAULT_FSHIN_LENGTH} 0 {-.09 * fshin_length/DEFAULT_FSHIN_LENGTH}" rgba="0.9 0.6 0.6 1" size="0.046 {fshin_length}" type="capsule"/>
          <body name="ffoot" pos="{.13 * fshin_length/DEFAULT_FSHIN_LENGTH} 0 {-.18 * fshin_length/DEFAULT_FSHIN_LENGTH}">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos="{.045 * ffoot_length/DEFAULT_FFOOT_LENGTH} 0 {-.07 * ffoot_length/DEFAULT_FFOOT_LENGTH}" rgba="0.9 0.6 0.6 1" size="0.046 {ffoot_length}" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="{bthigh_gear}" joint="bthigh" name="bthigh"/>
    <motor gear="{bshin_gear}" joint="bshin" name="bshin"/>
    <motor gear="{bfoot_gear}" joint="bfoot" name="bfoot"/>
    <motor gear="{fthigh_gear}" joint="fthigh" name="fthigh"/>
    <motor gear="{fshin_gear}" joint="fshin" name="fshin"/>
    <motor gear="{ffoot_gear}" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>'''
        # Write to a uniquely-named temp file per reset so vectorized workers never
        # trample each other's XML and use atomic replace to avoid partially-written files.
        fd, temp_path = tempfile.mkstemp(dir=self.xml_path, suffix='.xml')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(file_string)
                f.flush()
                os.fsync(f.fileno())
            final_path = temp_path
        except Exception:
            # If writing fails, make sure we clean up the temporary file.
            try:
                os.remove(temp_path)
            finally:
                pass
            raise
        return final_path
