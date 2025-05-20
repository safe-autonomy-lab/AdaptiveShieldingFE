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
"""OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from rich.progress import track
import jax
import numpy as np
from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.shield.vectorized_shield import VectorizedShield


class OnPolicyAdapter(OnlineAdapter):
    """OnPolicy Adapter for OmniSafe.

    :class:`OnPolicyAdapter` is used to adapt the environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._reset_log()

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
        shield: VectorizedShield = None,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        self._reset_log()

        obs, info = self.reset()
        if shield is not None:
            obs_dims = info['obs_dims'] if isinstance(info['obs_dims'], dict) else info['obs_dims'][0]['robot']
            shield.obs_dims = obs_dims
            shield._setup_environment_params()
            agent_pos, agent_mat = shield._process_agent_information(info)
            shield.prepare_dp_input(info['original_obs'], agent_pos, agent_mat, device=self._device)
            shield.agent_pos = agent_pos   
            shield.agent_mat = agent_mat
            xs_history = []
            ys_history = []
            if not hasattr(shield, "_compute_coefs"):
                shield._compute_coefs = shield.compute_coefficients_fn()
            
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            if shield is not None:
                obs = shield.add_coefficients_to_obs(obs)
                step_condition = shield.N < shield.example_nbr + 1
                one_step_after_condition = shield.prev_dp_input is not None
                # if self.warm_up_condition and step_condition and one_step_after_condition:
                agent_pos, agent_mat = shield._process_agent_information(info)
                if step_condition and one_step_after_condition:
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
                shield.prepare_dp_input(original_obs, agent_pos, agent_mat, device=self._device)

            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            if shield is not None:
                shield.prev_action = act
            
            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    ### manually set
                    if not done:
                        ### manually set
                        if epoch_end:
                            if isinstance(obs, dict):
                                obs_dict = {}
                                for key, item in obs.items():
                                    obs_dict[key] = item[idx][None]
                                _, last_value_r, last_value_c, _ = agent.step(obs_dict)
                                last_value_r = last_value_r.squeeze(0)
                                last_value_c = last_value_c.squeeze(0)
                            else:
                                _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                            
                        if time_out:
                            if isinstance(obs, dict):
                                obs_dict = {}
                                for key, item in info['final_observation'].items():
                                    obs_dict[key] = item[idx][None]
                                _, last_value_r, last_value_c, _ = agent.step(obs_dict)
                                last_value_r = last_value_r.squeeze(0)
                                last_value_c = last_value_c.squeeze(0)
                            else:
                                _, last_value_r, last_value_c, _ = agent.step(
                                    info['final_observation'][idx],
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                        self._ep_success[idx] = 0.0
                        self._ep_collision_cost[idx] = 0.0
                        self._ep_discomfort_cost[idx] = 0.0

                    buffer.finish_path(last_value_r, last_value_c, idx)

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
            self._ep_success = torch.zeros(self._env.num_envs)
            self._ep_collision_cost = torch.zeros(self._env.num_envs)
            self._ep_discomfort_cost = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
            self._ep_success[idx] = 0.0
            self._ep_collision_cost[idx] = 0.0
            self._ep_discomfort_cost[idx] = 0.0
