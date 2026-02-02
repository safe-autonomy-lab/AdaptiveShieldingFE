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
import glob
import warnings
from typing import Any, Dict
from omnisafe.models.actor_critic.constraint_actor_q_and_v_critic import ConstraintActorQAndVCritic

import numpy as np
import torch
from gymnasium.spaces import Box
import time
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
from omnisafe.envs.core import CMDP
from omnisafe.envs.wrapper import ActionScale, TimeLimit
from omnisafe.models.actor_critic import ConstraintActorCritic, ConstraintActorQCritic
from omnisafe.models.base import Actor
from omnisafe.utils.config import Config
from shield.vectorized_shield import VectorizedShield

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
        safety_budget: torch.Tensor | None = None,
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

        self._safety_budget: torch.Tensor = safety_budget
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
        if hasattr(self._env, "get_obs_normalizer"):
            self.normalizer = self._env.get_obs_normalizer()
        else:
            self.normalizer = None
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
        self.env_id = self._cfgs['env_id']
        if self._dict_cfgs.get('env_cfgs') is not None:
            env_kwargs.update(self._dict_cfgs['env_cfgs'])

        self.__load_model_and_env(env)
        self._load_shield_normalizer(save_dir)

    def _load_shield_normalizer(self, save_dir: str) -> None:
        if self.shield is None or not hasattr(self.shield, "normalizer"):
            return
        candidates = glob.glob(os.path.join(save_dir, "shield_normalizer.pt"))
        if not candidates:
            return
        best_epoch = -1
        best_path = None
        for path in candidates:
            name = os.path.basename(path)
            try:
                epoch = int(name.split("-")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = path
        if best_path is None:
            return
        payload = torch.load(best_path, map_location=self._device)
        state_dict = payload.get("state_dict", payload)
        try:
            self.shield.normalizer.load_state_dict(state_dict)
        except Exception as exc:  # pragma: no cover - defensive for mismatched shapes
            warnings.warn(f"Failed to load shield normalizer from {best_path}: {exc}")

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
        
        shield = self.shield
        for episode in range(num_episodes):
            self.frames = []
            obs, info = self._env.reset()
            episode_step = 0
            self.shield_triggered_count = 0
            if shield is not None:
                if hasattr(shield, "example_nbr") and hasattr(shield, "prediction_horizon"):
                    shield.example_nbr = min(shield.example_nbr, shield.prediction_horizon + 1)
                if hasattr(self._env, 'get_slices'):
                    robot_slices = self._env.get_slices()['robot']
                else:
                    robot_slices = info['robot'][0]
                action_low = torch.from_numpy(self._env.action_space.low).to(self._device).float()
                action_high = torch.from_numpy(self._env.action_space.high).to(self._device).float()
                if 'Velocity' in self._cfgs['env_id']:
                    agent_pos, agent_mat = info['x_velocity'].reshape(-1, 1), None
                else:
                    agent_pos, agent_mat = shield._process_agent_information(info)
                robot_original_obs = info['original_obs'][:, robot_slices] if 'original_obs' in info else obs[:, robot_slices]
                shield.prepare_dp_input(robot_original_obs, agent_pos, agent_mat, device=self._device)
                shield_trigger = False
                weights = torch.zeros(self.vector_env_nums, shield.n_basis).to(self._device)
                shield.coeffs_for_dynamics_prediction = weights
                shield.normalized_coeffs_for_dynamics_prediction = weights
                shield.robot_slices = robot_slices
                # Circle environment has a fixed sigwall location, so call it once here
                is_circle = ('Circle' in self._cfgs['env_id'])
                shield.is_circle = is_circle
                shield.range_limit = shield.static_threshold if is_circle else None
            self._safety_obs = torch.ones(1).unsqueeze(0)
            ep_ret, ep_cost, length = 0.0, 0.0, 0.0
            self.shield_trigger_count = 0
            self.episode_idx = episode
            self.current_time_step = 0

            start_time = time.time()
            

            done = False
            normalizer = self.normalizer
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
                            if 'Velocity' in self._cfgs['env_id']:
                                agent_pos = info['x_velocity'].reshape(-1, 1)
                                agent_mat = None
                            else:
                                agent_pos, agent_mat = shield._process_agent_information(info)

                            original_robot_obs = info['original_obs'][:, robot_slices] if 'original_obs' in info else obs[:, robot_slices]
                            shield.prepare_dp_input(original_robot_obs, agent_pos, agent_mat, device=self._device)
                            if shield.prev_dp_input is not None and shield.prev_action is not None:
                                shield_trigger = shield.update_weights(episode_step)
                                obs[:, -shield.n_basis:] = shield.normalized_coeffs_for_dynamics_prediction
                                
                            act = self._get_shielded_actions(obs, info, self._actor, self.shield, action_low, action_high, shield_trigger).reshape(1, -1)

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

                episode_step += 1

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
                if done and shield is not None:
                    shield.reset()
                    episode_step = 0
                    shield_trigger = False

            end_time = time.time()
            episode_run_times.append(end_time - start_time)
            episode_rewards.append(ep_ret)
            episode_costs.append(ep_cost)
            episode_lengths.append(length)
            shield_trigger_counts.append(self.shield_triggered_count)
            episode_hidden_parameters.append(tuple(info['hidden_parameters_features'][0].ravel()))
            print(f'Episode {episode} results:')
            print(f'Episode reward: {ep_ret}')
            print(f'Episode cost: {ep_cost}')
            print(f'Episode length: {length}')
            print(f'Shield triggered: {self.shield_triggered_count}')
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
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        shield_trigger: bool,
    ) -> torch.Tensor:
        """Enhanced shielded action selection with safety memory for vectorized environments."""
        # Check presafety condition
        unsafe_mask = shield._check_presafety_condition(info, enhanced_safety=0.15)
        unsafe_mask = torch.from_numpy(unsafe_mask).to(self._device)
        dones = torch.zeros(self.vector_env_nums, device=self._device).bool()
        # We save robot's position only because conformal prediction is based on robot's position
        shield.update_robot_actual_history(shield.dp_y)
        shield.step_last_triggered += 1
        if shield.shield_triggered:
            # This way, we make sure the conformal prediction scores are correctly updated, time step matching.
            shield.update_conformality_scores()
            shield._set_conformal_thresholds()
            shield.shield_triggered = False
        
        if shield.step_last_triggered > shield.idle_condition and shield_trigger and shield.prediction_horizon > 0 and unsafe_mask.any():
            agent_pos, _, buttons, goals, hazards, vases, pillars, push_boxes, gremlins, circle = shield.process_info(info, self.vector_env_nums)
            with torch.no_grad():
                acts, value_r, value_c, logps = agent.sample(obs_tensor_for_policy, n_samples=shield.sampling_nbr, scale=shield.scale)
                action_clipped = acts.clamp(action_low, action_high)
            
            is_safe, min_indices, safety_measure = shield.sample_safe_actions(
                shield.dp_input,
                agent_pos,
                buttons,
                goals,
                hazards,
                vases,
                pillars,
                push_boxes,
                gremlins,
                circle,
                first_action=action_clipped,
                device=self._device,
                selection_method='top-k',
                k=max(shield.sampling_nbr // 5, 1),
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
            logp = logps[min_indices, np.arange(len(min_indices))]
            
            self.shield_triggered_count += 1
            shield.shield_triggered = True
            shield.step_last_triggered = 0
        else:
            act, value_r, value_c, logp = agent.step(obs_tensor_for_policy)
            shield.shield_triggered = False

        shield.prev_action = act
        return act
