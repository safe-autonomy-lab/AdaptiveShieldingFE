# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of Evaluator."""

from __future__ import annotations

import json
import os
import warnings
from functools import partial
from typing import Any, Tuple, Dict
from omnisafe.models.actor_critic.constraint_actor_q_and_v_critic import ConstraintActorQAndVCritic

import numpy as np
import torch
import jax
from gymnasium.spaces import Box
from gymnasium.utils.save_video import save_video
from torch import nn
import time
import matplotlib.pyplot as plt
from PIL import Image

from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.planner import (
    ARCPlanner,
    CAPPlanner,
    CCEPlanner,
    CEMPlanner,
    RCEPlanner,
    SafeARCPlanner,
)
from omnisafe.common import Normalizer
from omnisafe.common.control_barrier_function.crabs.models import (
    AddGaussianNoise,
    CrabsCore,
    ExplorationPolicy,
    MeanPolicy,
    MultiLayerPerceptron,
)
from omnisafe.common.control_barrier_function.crabs.optimizers import Barrier
from omnisafe.common.control_barrier_function.crabs.utils import Normalizer as CRABSNormalizer
from omnisafe.common.control_barrier_function.crabs.utils import create_model_and_trainer
from omnisafe.envs.core import CMDP, make
from omnisafe.envs.wrapper import ActionRepeat, ActionScale, ObsNormalize, TimeLimit
from omnisafe.models.actor import ActorBuilder
from omnisafe.models.actor_critic import ConstraintActorCritic, ConstraintActorQCritic
from omnisafe.models.base import Actor
from omnisafe.utils.config import Config
from omnisafe.shield.vectorized_shield import VectorizedShield

t2numpy = lambda x: x.cpu().detach().numpy()

class Evaluator:  # pylint: disable=too-many-instance-attributes
    """This class includes common evaluation methods for safe RL algorithms.

    Args:
        env (CMDP or None, optional): The environment. Defaults to None.
        actor (Actor or None, optional): The actor. Defaults to None.
        render_mode (str, optional): The render mode. Defaults to 'rgb_array'.
    """

    _cfgs: Config
    _dict_cfgs: dict[str, Any]
    _save_dir: str
    _model_name: str
    _cost_count: torch.Tensor

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        env: CMDP | None = None,
        unwrapped_env: CMDP | None = None,
        actor: Actor | None = None,
        actor_critic: ConstraintActorCritic | ConstraintActorQCritic | None = None,
        dynamics: EnsembleDynamicsModel | None = None,
        planner: (
            CEMPlanner | ARCPlanner | SafeARCPlanner | CCEPlanner | CAPPlanner | RCEPlanner | None
        ) = None,
        shield: VectorizedShield | None = None,
        render_mode: str = 'rgb_array',
    ) -> None:
        """Initialize an instance of :class:`Evaluator`."""
        self._env: CMDP | None = env
        self._unwrapped_env: CMDP | None = unwrapped_env
        self._actor: Actor | None = actor
        self._actor_critic: ConstraintActorCritic | ConstraintActorQCritic | ConstraintActorQAndVCritic | None = actor_critic
        self._dynamics: EnsembleDynamicsModel | None = dynamics
        self._planner = planner
        self._dividing_line: str = '\n' + '#' * 50 + '\n'
        self.shield = shield
        self.vector_env_nums = 1
        self._device = torch.device('cpu')

        self._safety_budget: torch.Tensor
        self._safety_obs = torch.ones(1)
        self._cost_count = torch.zeros(1)
        self.__set_render_mode(render_mode)

    def __set_render_mode(self, render_mode: str) -> None:
        """Set the render mode.

        Args:
            render_mode (str, optional): The render mode. Defaults to 'rgb_array'.

        Raises:
            NotImplementedError: If the render mode is not implemented.
        """
        # set the render mode
        if render_mode in ['human', 'rgb_array', 'rgb_array_list']:
            self._render_mode: str = render_mode
        else:
            raise NotImplementedError('The render mode is not implemented.')

    def __load_cfgs(self, save_dir: str) -> None:
        """Load the config from the save directory.

        Args:
            save_dir (str): Directory where the model is saved.

        Raises:
            FileNotFoundError: If the config file is not found.
        """
        cfg_path = os.path.join(save_dir, 'config.json')
        try:
            with open(cfg_path, encoding='utf-8') as file:
                kwargs = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'The config file is not found in the save directory{save_dir}.',
            ) from error
        self._dict_cfgs = kwargs
        self._cfgs = Config.dict2config(kwargs)

    # pylint: disable-next=too-many-branches
    def __load_model_and_env(
        self,
        env,
    ) -> None:
        """Load the model from the save directory.

        Args:
            save_dir (str): Directory where the model is saved.
            model_name (str): Name of the model.
            env_kwargs (dict[str, Any]): Keyword arguments for the environment.

        Raises:
            FileNotFoundError: If the model is not found.
        """
        # load the saved model
        self._env = env
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        assert isinstance(observation_space, Box), 'The observation space must be Box.'
        assert isinstance(action_space, Box), 'The action space must be Box.'
        self.normalizer = self._env.get_obs_normalizer()
        time_limit = env.max_episode_steps
        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, device=torch.device('cpu'), time_limit=time_limit)
        self._env = ActionScale(self._env, device=torch.device('cpu'), low=-1.0, high=1.0)

    # pylint: disable-next=too-many-locals
    def load_saved(
        self,
        save_dir: str,
        render_mode: str = 'rgb_array',
        camera_name: str | None = None,
        camera_id: int | None = None,
        width: int = 256,
        height: int = 256,
        env: CMDP | None = None,
    ) -> None:
        """Load a saved model.

        Args:
            save_dir (str): The directory where the model is saved.
            model_name (str): The name of the model.
            render_mode (str, optional): The render mode, ranging from 'human', 'rgb_array',
                'rgb_array_list'. Defaults to 'rgb_array'.
            camera_name (str or None, optional): The name of the camera. Defaults to None.
            camera_id (int or None, optional): The id of the camera. Defaults to None.
            width (int, optional): The width of the image. Defaults to 256.
            height (int, optional): The height of the image. Defaults to 256.
        """
        # load the config
        self._save_dir = save_dir
        self.__load_cfgs(save_dir)

        self.__set_render_mode(render_mode)

        env_kwargs = {
            'env_id': self._cfgs['env_id'],
            'num_envs': 1,
            'render_mode': self._render_mode,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'width': width,
            'height': height,
        }
        if self._dict_cfgs.get('env_cfgs') is not None:
            env_kwargs.update(self._dict_cfgs['env_cfgs'])

        self.__load_model_and_env(env)

    @property
    def fps(self) -> int:
        """The fps of the environment.

        Raises:
            AssertionError: If the environment is not provided or created.
            AtrributeError: If the fps is not found.
        """
        assert (
            self._env is not None
        ), 'The environment must be provided or created before getting the fps.'
        try:
            fps = self._env.metadata['render_fps']
        except (AttributeError, KeyError):
            fps = 30
            warnings.warn('The fps is not found, use 30 as default.', stacklevel=2)

        return fps

    def save_frames(self, start_frame: int = 0, end_frame: int = -1) -> None:
        """Save frames as GIF for the current episode."""
        if not self.frames:
            return
            
        env_id = self._cfgs['env_id']
        # Create evaluation_plot directory if it doesn't exist
        plot_dir = os.path.join('evaluation_plot', env_id)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Convert frames to PIL Images
        pil_images = [Image.fromarray(frame) for frame in self.frames[start_frame: end_frame]]
        
        # Save GIF
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(plot_dir, f'episode_{self.episode_idx}_{timestamp}.gif')
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=1000 // self.fps,  # Duration in milliseconds
            loop=0  # 0 means loop forever
        )
        
        # Clear frames for next episode
        self.frames = []
    
    def evaluate(
        self,
        num_episodes: int = 10,
        cost_criteria: float = 1.0,
        save_plot: bool = False,
        seed: int = 0,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        """Evaluate the agent for num_episodes episodes.

        Args:
            num_episodes (int, optional): The number of episodes to evaluate. Defaults to 10.
            cost_criteria (float, optional): The cost criteria. Defaults to 1.0.

        Returns:
            (episode_rewards, episode_costs): The episode rewards and costs.

        Raises:
            ValueError: If the environment and the policy are not provided or created.
        """
        if self._env is None or (self._actor is None and self._planner is None):
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.',
            )
        import random
        import time
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._env.set_seed(seed)
        episode_rewards: list[float] = []
        episode_costs: list[float] = []
        episode_lengths: list[float] = []
        episode_run_times: list[float] = []
        shield_trigger_counts: list[float] = []
        episode_hidden_parameters: list[list[float]] = []
        if self.shield is not None:
            if not hasattr(self.shield, "_compute_coefs"):
                self.shield._compute_coefs = self.shield.compute_coefficients_fn()
        
        shield = self.shield
        for episode in range(num_episodes):
            self.frames = []
            obs, info = self._env.reset()
            start_time = time.time()
            self._env.set_episode_count(episode)
            self._safety_obs = torch.ones(1)
            ep_ret, ep_cost, length = 0.0, 0.0, 0.0
            self.shield_trigger_count = 0
            self.episode_idx = episode
            self.current_time_step = 0
            xs_history = []
            ys_history = []

            done = False
            while not done:
                if save_plot:
                    frame = self._env.render()
                    if isinstance(frame, tuple):
                        frame = frame[0]
                    self.frames.append(frame)
                
                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    obs = torch.cat([obs, self._safety_obs], dim=-1)

                with torch.no_grad():
                    if self._actor is not None:
                        if self.shield is not None:
                            one_step_after_condition = self.shield.prev_dp_input is not None
                            step_condition = self.shield.N < self.shield.example_nbr + 1
                            agent_pos, agent_mat = self.shield._process_agent_information(info)
                            if shield.use_fe_representation and one_step_after_condition and step_condition:
                                obs = shield.add_coefficients_to_obs(obs)
                                example_x = torch.cat([shield.prev_dp_input, shield.prev_action], axis=-1).cpu().detach().numpy()
                                example_y = agent_pos

                                xs_history.append(example_x[:, None, :])
                                ys_history.append(example_y[:, None, :])
                                
                                example_xs = np.concatenate(xs_history, axis=1)
                                example_ys = np.concatenate(ys_history, axis=1)

                                example_xs = jax.device_put(example_xs)
                                example_ys = jax.device_put(example_ys)

                                new_ws = shield._compute_coefs(shield.dp_state, example_xs, example_ys)
                                shield.update_ws(new_ws)
                                shield.N += 1

                                if shield.N == shield.example_nbr + 1:
                                    ws = shield.ws.copy()
                                    shield.ws_representation = ws * shield.scale / np.linalg.norm(ws, axis=1).reshape(-1, 1)        
                                    xs_history = []
                                    ys_history = []

                            original_obs = info['original_obs']
                            self.shield.prepare_dp_input(original_obs, agent_pos, agent_mat, device=self._device)
                            act = self._get_shielded_actions(obs, info, self._actor, self.shield, self.normalizer, length >= self.shield.max_history - 1, save_plot)
                        else:
                            act = self._actor.predict(
                                obs.reshape(
                                    -1,
                                obs.shape[-1],  # to make sure the shape is (1, obs_dim)
                            )[None],
                            deterministic=True,
                        ).reshape(
                            1, -1,  # to make sure the shape is (act_dim,)
                        )
                    elif self._planner is not None:
                        act = self._planner.output_action(
                            obs.unsqueeze(0).to('cpu'),
                        )[
                            0
                        ].squeeze(0)
                    else:
                        raise ValueError(
                            'The policy must be provided or created before evaluating the agent.',
                        )
                    
                obs, rew, cost, terminated, truncated, info = self._env.step(act)
                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    self._safety_obs -= cost.unsqueeze(-1) / self._safety_budget
                    self._safety_obs /= self._cfgs.algo_cfgs.saute_gamma

                ep_ret += rew.item()
                ep_cost += (cost_criteria**length) * cost.item()
                if (
                    'EarlyTerminated' in self._cfgs['algo']
                    and ep_cost >= self._cfgs.algo_cfgs.cost_limit
                ):
                    terminated = torch.as_tensor(True)
                length += 1
                self.current_time_step = length

                done = bool(terminated or truncated)

            end_time = time.time()
            episode_run_times.append(end_time - start_time)
            episode_rewards.append(ep_ret)
            episode_costs.append(ep_cost)
            episode_lengths.append(length)
            shield_trigger_counts.append(self.shield_trigger_count)
            episode_hidden_parameters.append(tuple(info['hidden_parameters'][0].ravel()))
            print(f'Episode {episode} results:')
            print(f'Episode reward: {ep_ret}')
            print(f'Episode cost: {ep_cost}')
            print(f'Episode length: {length}')
            print(f'Shield triggered: {self.shield_trigger_count}')
            print(f'Episode run time: {end_time - start_time}')
            if save_plot:
                self.save_frames(start_frame=350, end_frame=750)

        print(self._dividing_line)
        print('Evaluation results:')
        print(f'Average episode reward: {np.mean(a=episode_rewards)}')
        print(f'Average episode cost: {np.mean(a=episode_costs)}')
        print(f'Average episode length: {np.mean(a=episode_lengths)}')
        print(f'Average shield triggered: {np.mean(a=shield_trigger_counts)}')
        print(f'Average episode run time: {np.mean(a=episode_run_times)}')
        self._env.close()
        return (
            episode_rewards,
            episode_costs,
            episode_lengths,
            shield_trigger_counts,
            episode_run_times,
            episode_hidden_parameters,
        )

    @property
    def fps(self) -> int:
        """The fps of the environment.

        Raises:
            AssertionError: If the environment is not provided or created.
            AtrributeError: If the fps is not found.
        """
        assert (
            self._env is not None
        ), 'The environment must be provided or created before getting the fps.'
        try:
            fps = self._env.metadata['render_fps']
        except (AttributeError, KeyError):
            fps = 30
            warnings.warn('The fps is not found, use 30 as default.', stacklevel=2)

        return fps

    def _get_shielded_actions(
        self,
        obs_tensor_for_policy: torch.Tensor,
        info: Dict,
        agent: ConstraintActorQAndVCritic,
        shield: VectorizedShield,
        normalizer: Normalizer,
        shield_step_condition: bool,
        save_plot: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced shielded action selection with safety memory for vectorized environments."""
        # Check presafety condition
        unsafe_mask = shield._check_presafety_condition(info, enhanced_safety=0.15)
        unsafe_mask = torch.from_numpy(unsafe_mask).to(self._device)
        dones = torch.zeros(self.vector_env_nums, device=self._device).bool()

        if not shield_step_condition:
            agent_pos, agent_mat, goal_pos, hazards, pillars, gremlins = shield.process_info(info, self.vector_env_nums)
        
        # if self.warm_up_condition and unsafe_mask.any() and shield_step_condition:
        if unsafe_mask.any() and shield_step_condition:
            agent_pos, agent_mat, goal_pos, hazards, pillars, gremlins = shield.process_info(info, self.vector_env_nums)
            dp_input = t2numpy(shield.dp_input)
            shield.update_robot_actual_history(agent_pos)
            dp_acp_region = shield.robot_conformal_threshold
            dp_acp_region = min(dp_acp_region, 0.1)
            iter_nbr = 0
            max_iter_nbr = 1
            while not dones.all() and iter_nbr < max_iter_nbr:
                if save_plot:
                    env_id = self._cfgs['env_id']
                    # Create evaluation_plot directory if it doesn't exist
                    plot_dir = os.path.join('evaluation_plot', env_id)
                    os.makedirs(plot_dir, exist_ok=True)
                    frame = self._unwrapped_env.render_rgb_array()
                    # Handle different render output types
                    frame_path = os.path.join(plot_dir, f'frame_{self.episode_idx}_{self.current_time_step}.png')
                    plt.imsave(frame_path, frame)
                    
                acts = agent.predict_with_multiple_samples_with_seed(obs_tensor_for_policy, n_samples=shield.sampling_nbr, seed=self.episode_idx)
                action_clipped = np.clip(t2numpy(acts), self._env.action_space.low, self._env.action_space.high)
                fixed_seed_policy = partial(agent.predict_with_seed, seed=self.episode_idx)
                
                is_safe, min_indices, safety_measure = shield.sample_safe_actions(
                    dp_input,
                    agent_pos,
                    agent_mat,
                    goal_pos,
                    hazards,
                    pillars,
                    gremlins,
                    first_action=action_clipped,
                    policy=fixed_seed_policy,
                    dp_acp_region=dp_acp_region,
                    device=self._device,
                    selection_method='top-k',
                    # selection_method='greedy',
                    k=max(shield.sampling_nbr // 10, 1),
                    normalizer=normalizer,
                )
                # Save action_clipped and safety_measure with timestamp
                if False:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_path = f"shield_data_{timestamp}.npz"
                    np.savez(
                        save_path,
                        action_clipped=action_clipped,
                        safety_measure=safety_measure
                    )
                dones = is_safe | dones    
                act = acts[min_indices, np.arange(len(min_indices))]
                
                if iter_nbr == 0:
                    final_actions = act.clone()
                else:
                    final_actions[dones] = act[dones]

                self.shield_trigger_count += 1
                iter_nbr += 1
            shield.update_conformality_scores()
            shield._set_conformal_thresholds()
        else:
            act = agent.predict(obs_tensor_for_policy)
            shield.shield_triggered = False

        shield.prev_action = act
        return act