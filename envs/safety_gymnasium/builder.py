# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Env builder."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import gymnasium
import numpy as np

from envs.safety_gymnasium import tasks
from envs.safety_gymnasium.bases.base_task import BaseTask
from envs.safety_gymnasium.utils.common_utils import ResamplingError, quat2zalign
from envs.safety_gymnasium.utils.task_utils import get_task_class_name


@dataclass
class RenderConf:
    r"""Render options.

    Attributes:
        mode (str): render mode, can be 'human', 'rgb_array', 'depth_array'.
        width (int): width of the rendered image.
        height (int): height of the rendered image.
        camera_id (int): camera id to render.
        camera_name (str): camera name to render.

        Note:
            ``camera_id`` and ``camera_name`` can only be set one of them.
    """

    mode: str = None
    width: int = 256
    height: int = 256
    camera_id: int = None
    camera_name: str = None


# pylint: disable-next=too-many-instance-attributes
class Builder(gymnasium.Env, gymnasium.utils.EzPickle):
    r"""An entry point to organize different environments, while showing unified API for users.

    The Builder class constructs the basic control framework of environments, while
    the details were hidden. There is another important parts, which is **task module**
    including all task specific operation.

    Methods:

    - :meth:`_setup_simulation`: Set up mujoco the simulation instance.
    - :meth:`_get_task`: Instantiate a task object.
    - :meth:`set_seed`: Set the seed for the environment.
    - :meth:`reset`: Reset the environment.
    - :meth:`step`: Step the environment.
    - :meth:`_reward`: Calculate the reward.
    - :meth:`_cost`: Calculate the cost.
    - :meth:`render`: Render the environment.

    Attributes:

    - :attr:`task_id` (str): Task id.
    - :attr:`config` (dict): Pre-defined configuration of the environment, which is passed via
      :meth:`safety_gymnasium.register()`.
    - :attr:`render_parameters` (RenderConf): Render parameters.
    - :attr:`action_space` (gymnasium.spaces.Box): Action space.
    - :attr:`observation_space` (gymnasium.spaces.Dict): Observation space.
    - :attr:`obs_space_dict` (dict): Observation space dictionary.
    - :attr:`done` (bool): Whether the episode is done.
    """

    metadata: ClassVar[dict[str, Any]] = {
        'render_modes': [
            'human',
            'rgb_array',
            'depth_array',
        ],
        'render_fps': 30,
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        task_id: str,
        config: dict | None = None,
        render_mode: str | None = None,
        width: int = 256,
        height: int = 256,
        camera_id: int | None = None,
        camera_name: str | None = None,
        env_config: dataclass | None = None,
    ) -> None:
        """Initialize the builder.

        Note:
            The ``camera_name`` parameter can be chosen from:
              - **human**: The camera used for freely moving around and can get input
                from keyboard real time.
              - **vision**: The camera used for vision observation, which is fixed in front of the
                agent's head.
              - **track**: The camera used for tracking the agent.
              - **fixednear**: The camera used for top-down observation.
              - **fixedfar**: The camera used for top-down observation, but is further than **fixednear**.

        Args:
            task_id (str): Task id.
            config (dict): Pre-defined configuration of the environment, which is passed via
              :meth:`safety_gymnasium.register`.
            render_mode (str): Render mode, can be 'human', 'rgb_array', 'depth_array'.
            width (int): Width of the rendered image.
            height (int): Height of the rendered image.
            camera_id (int): Camera id to render.
            camera_name (str): Camera name to render.
        """
        gymnasium.utils.EzPickle.__init__(self, config=config)
        assert env_config is not None, 'env_config must be provided'
        self.task_id: str = task_id
        self.config: dict = config
        self.env_config = env_config
        self.fix_hidden_parameters: bool = env_config.FIX_HIDDEN_PARAMETERS
        
        self._seed: int = None
        self.obs_dims = None
        self._setup_simulation()

        self.first_reset: bool = None
        self.steps: int = None
        self.cost: float = None
        self.terminated: bool = True
        self.truncated: bool = False

        self.render_parameters = RenderConf(render_mode, width, height, camera_id, camera_name)

    def _setup_simulation(self) -> None:
        """Set up mujoco the simulation instance."""
        self.task = self._get_task()
        self.set_seed()

    def get_obs_dims(self):
        self.obs_dims = {}
        obs_dims = 0
        for v in self.task.agent.obs_sensor().values():
            obs_dims += v.flatten().shape[0]

        self.obs_dims['robot'] = obs_dims
        for obstacle in self.task._obstacles:
            if obstacle.is_lidar_observed:
                name = obstacle.name
            if obstacle.pos is None:
                continue
            self.obs_dims[name] = len(self.task._obs_lidar(obstacle.pos, obstacle.group))        
        self.obs_dims['hidden_parameters'] = self.env_config.NBR_OF_HIDDEN_PARAMETERS
        return self.obs_dims
    
    def get_episode_count(self):
        return self.task.world._episode_count

    def set_episode_count(self, episode_count: int):
        self.task.world._episode_count = episode_count

    def _get_task(self) -> BaseTask:
        """Instantiate a task object."""
        class_name = self.config.get('task_name', get_task_class_name(self.task_id))
        assert hasattr(tasks, class_name), f'Task={class_name} not implemented.'
        task_class = getattr(tasks, class_name)
        task = task_class(config=self.config)

        task.build_observation_space()
        task.set_env_config(self.env_config)
        return task

    def set_seed(self, seed: int | None = None) -> None:
        """Set internal random state seeds."""
        self._seed = np.random.randint(2**32, dtype='int64') if seed is None else seed
        self.task.random_generator.set_random_seed(self._seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:  # pylint: disable=arguments-differ
        """Reset the environment and return observations."""
        info = {}

        if not self.task.mechanism_conf.randomize_layout:
            assert seed is None, 'Cannot set seed if randomize_layout=False'
            self.set_seed(0)
        elif seed is not None:
            self.set_seed(seed)

        self.terminated = False
        self.truncated = False
        self.steps = 0  # Count of steps taken in this episode

        self.task.reset()
        self.task.specific_reset()
        self.task.update_world()  # refresh specific settings
        
        self.task.agent.reset()

        cost = self._cost()
        assert cost['cost_sum'] == 0, f'World has starting cost! {cost}'
        # Reset stateful parts of the environment
        self.first_reset = False  # Built our first world successfully
        self.hidden_parameters = self.task.world.hidden_parameters
        info['hidden_parameters'] = self.hidden_parameters
        objects_pos, min_distance = self._get_objects_pos_obses()
        info.update(objects_pos)
        info['min_distance'] = min_distance

        if hasattr(self.task, 'circle'):
            info['circle_rad'] = self.task.circle.radius
            info['sigwalls_loc'] = self.task.sigwalls.locate_factor

        info['is_success'] = self.task.goal_achieved
        info['cost'] = cost
        
        if hasattr(self.task, 'agent_vel'):
            vel = self.task.agent_vel()
            info['agent_vel'] = vel
            velocity_constraint = 1.0
            if vel > velocity_constraint:
                cost = 1.
        
        if hasattr(self.task, 'goal'):
            info['goal_pos'] = self.task.goal.pos[np.newaxis, :2]
            info['dist2goal'] = self.task.dist_goal()

        self.get_obs_dims() if self.obs_dims is None else None
        info['obs_dims'] = self.obs_dims
        obs = self.task.obs()
        return (obs, info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, float, bool, bool, dict]:
        """Take a step and return observation, reward, cost, terminated, truncated, info."""
        assert not self.done, 'Environment must be reset before stepping.'
        action = np.array(action, copy=False)  # cast to ndarray
        if action.shape != self.action_space.shape:  # check action dimension
            raise ValueError('Action dimension mismatch')

        info = {}
        info['hidden_parameters'] = self.hidden_parameters

        exception = self.task.simulation_forward(action)
        if exception:
            self.truncated = True

            reward = self.task.reward_conf.reward_exception
            info['cost_exception'] = 1.0
        else:
            # Reward processing
            reward = self._reward()
            # Constraint violations
            info.update(self._cost())
            cost = info['cost_sum']
            self.task.specific_step()
            # Goal processing
            if self.task.goal_achieved:
                info['goal_met'] = True
                if self.task.mechanism_conf.continue_goal:
                    # Update the internal layout
                    # so we can correctly resample (given objects have moved)
                    self.task.update_layout()
                    # Try to build a new goal, end if we fail
                    if self.task.mechanism_conf.terminate_resample_failure:
                        try:
                            self.task.update_world()
                        except ResamplingError:
                            # Normal end of episode
                            self.terminated = True
                    else:
                        # Try to make a goal, which could raise a ResamplingError exception
                        self.task.update_world()
                else:
                    self.terminated = True

        # termination of death processing
        if not self.task.agent.is_alive():
            self.terminated = True

        # Timeout
        self.steps += 1
        if self.steps >= self.task.num_steps:
            self.truncated = True  # Maximum number of steps in an episode reached

        if self.render_parameters.mode == 'human':
            self.render()
        
        objects_pos, min_distance = self._get_objects_pos_obses()
        if hasattr(self.task, 'circle'):
            info['circle_rad'] = self.task.circle.radius
            info['sigwalls_loc'] = self.task.sigwalls.locate_factor

        info.update(objects_pos)
        info['is_success'] = self.task.goal_achieved
        info['min_distance'] = min_distance
        # this is only for run environment
        if hasattr(self.task, 'agent_vel'):
            vel = self.task.agent_vel()
            info['agent_vel'] = vel
            velocity_constraint = 1.5
            if vel > velocity_constraint:
                cost = 1.
        
        if hasattr(self.task, 'goal'):
            info['goal_pos'] = self.task.goal.pos[np.newaxis, :2]
            info['dist2goal'] = self.task.dist_goal()
        
        obs = self.task.obs()
        return (obs, reward, cost, self.terminated, self.truncated, info)
    
    def _get_objects_pos_obses(self):
        objects_pos = {}
        min_distance = np.inf
        agent_pos = self.task.agent.pos

        objects_pos['agent_pos'] = agent_pos
        objects_pos['agent_mat'] = self.task.agent.mat.flatten()

        agent_xy_pos = agent_pos[:2]
        
        for obstacle in self.task._obstacles:
            obs_pos = obstacle.pos
            obs_name = obstacle.name
            if obs_pos is None:
                continue

            if obs_name not in ['hazards', 'pillars', 'gremlins', 'goal']:
                continue            
            
            # Handle both single position and list of positions
            if isinstance(obs_pos[0], np.ndarray):
                obj_pos = []
                for pos in obs_pos:
                    xy_pos = pos[:2]
                    min_distance = min(min_distance, np.linalg.norm(agent_xy_pos - xy_pos))
                    obj_pos.append(xy_pos)
                    
                objects_pos[f'{obs_name}_pos'] = np.stack(obj_pos, axis=0)
            else:
                # Single position case
                xy_pos = obs_pos[:2]
                min_distance = min(min_distance, np.linalg.norm(agent_xy_pos - xy_pos))        
        return objects_pos, min_distance

    def _reward(self) -> float:
        """Calculate the current rewards.

        Call exactly once per step.
        """
        reward = self.task.calculate_reward()

        # Intrinsic reward for uprightness
        if self.task.reward_conf.reward_orientation:
            zalign = quat2zalign(
                self.task.data.get_body_xquat(self.task.reward_conf.reward_orientation_body),
            )
            reward += self.task.reward_conf.reward_orientation_scale * zalign

        # Clip reward
        reward_clip = self.task.reward_conf.reward_clip
        if reward_clip:
            in_range = -reward_clip < reward < reward_clip
            if not in_range:
                reward = np.clip(reward, -reward_clip, reward_clip)
                print('Warning: reward was outside of range!')

        return reward

    def _cost(self) -> dict:
        """Calculate the current costs and return a dict.

        Call exactly once per step.
        """
        cost = self.task.calculate_cost()

        # Optionally remove shaping from reward functions.
        if self.task.cost_conf.constrain_indicator:
            for k in list(cost.keys()):
                cost[k] = float(cost[k] > 0.0)  # Indicator function

        self.cost = cost

        return cost

    def render(self) -> np.ndarray | None:
        """Call underlying :meth:`safety_gymnasium.bases.underlying.Underlying.render` directly.

        Width and height in parameters are constant defaults for rendering
        frames for humans. (not used for vision)

        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if render_mode is:

        - None (default): no render is computed.
        - human: render return None.
          The environment is continuously rendered in the current display or terminal. Usually for human consumption.
        - rgb_array: return a single frame representing the current state of the environment.
          A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
        - rgb_array_list: return a list of frames representing the states of the environment since the last reset.
          Each frame is a numpy.ndarray with shape (x, y, 3), as with `rgb_array`.
        - depth_array: return a single frame representing the current state of the environment.
          A frame is a numpy.ndarray with shape (x, y) representing depth values for an x-by-y pixel image.
        - depth_array_list: return a list of frames representing the states of the environment since the last reset.
          Each frame is a numpy.ndarray with shape (x, y), as with `depth_array`.
        """
        assert self.render_parameters.mode, 'Please specify the render mode when you make env.'
        assert (
            not self.task.observe_vision
        ), 'When you use vision envs, you should not call this function explicitly.'
        return self.task.render(cost=self.cost, **asdict(self.render_parameters))

    @property
    def action_space(self) -> gymnasium.spaces.Box:
        """Helper to get action space."""
        return self.task.action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Box | gymnasium.spaces.Dict:
        """Helper to get observation space."""
        return self.task.observation_space

    @property
    def obs_space_dict(self) -> dict[str, gymnasium.spaces.Box]:
        """Helper to get observation space dictionary."""
        return self.task.obs_info.obs_space_dict

    @property
    def done(self) -> bool:
        """Whether this episode is ended."""
        return self.terminated or self.truncated

    @property
    def render_mode(self) -> str:
        """The render mode."""
        return self.render_parameters.mode

    def __deepcopy__(self, memo) -> Builder:
        """Make class instance copyable."""
        other = Builder(
            self.task_id,
            self.config,
            self.render_parameters.mode,
            self.render_parameters.width,
            self.render_parameters.height,
            self.render_parameters.camera_id,
            self.render_parameters.camera_name,
        )
        other._seed = self._seed
        other.first_reset = self.first_reset
        other.steps = self.steps
        other.cost = self.cost
        other.terminated = self.terminated
        other.truncated = self.truncated
        other.task = deepcopy(self.task)  # pylint: disable=attribute-defined-outside-init
        return other
