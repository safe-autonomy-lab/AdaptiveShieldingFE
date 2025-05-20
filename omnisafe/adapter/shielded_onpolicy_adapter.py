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

from typing import Tuple, Dict

import torch
from rich.progress import track

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorShieldedOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.common.normalizer import Normalizer
from omnisafe.models.actor_critic.constraint_actor_q_and_v_critic import ConstraintActorQAndVCritic
from omnisafe.utils.config import Config
from omnisafe.shield.vectorized_shield import VectorizedShield
import jax
import numpy as np
import time

t2numpy = lambda x: x.cpu().detach().numpy()

class ShieldedOnPolicyAdapter(OnPolicyAdapter):
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
        """Initialize an instance of :class:`ShieldedOnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self.vector_env_nums = num_envs
        self.env_id = env_id
        self._reset_log()

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorQAndVCritic,
        buffer: VectorShieldedOnPolicyBuffer,
        logger: Logger,
        shield: VectorizedShield,
        normalizer: Normalizer = None,
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
        # self.safety_penalty_handler.register_safety_keys(logger)
        obs, info = self.reset()
        obs_dims = info['obs_dims'][0]['robot']
        shield.obs_dims = obs_dims    
        shield._setup_environment_params()
        agent_pos, agent_mat = shield._process_agent_information(info)
        shield.prepare_dp_input(info['original_obs'], agent_pos, agent_mat, device=self._device)
        shield.agent_pos = agent_pos
        shield.agent_mat = agent_mat
        
        self.shield_gradient_scale = shield.gradient_scale
        ep_shield_violation = torch.zeros(self.vector_env_nums, device=self._device)
        self.current_epoch = logger.current_epoch
        self.warm_up_epochs = shield.warm_up_epochs

        xs_history = []
        ys_history = []

        if not hasattr(shield, "_compute_coefs"):
            shield._compute_coefs = shield.compute_coefficients_fn()

        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {self.current_epoch}...',
        ):
            one_step_after_condition = shield.prev_dp_input is not None
            step_condition = shield.N < shield.example_nbr + 1
            # This condition is to update coefficients to infer hidden parameters            
            # if self.warm_up_condition and shield.use_fe_representation and one_step_after_condition and step_condition:
            agent_pos, agent_mat = shield._process_agent_information(info)
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
            shield.prepare_dp_input(original_obs, agent_pos, agent_mat, device=self._device)
            act, value_r, value_c, value_shield_violation, logp = self._get_shielded_actions(obs, info, agent, shield, normalizer, step >= shield.max_history - 1)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            
            unsafe_condition = torch.from_numpy(shield._check_presafety_condition(info, enhanced_safety=0.)).to(self._device)
            safety_violation = torch.logical_or(unsafe_condition, cost.to(self._device)).float()
            
            self._log_value(reward=reward, cost=cost, info=info)
            logger.store({'Safety/shield_violation': value_shield_violation.cpu()})

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
                shield_violation=safety_violation,
                value_shield_violation=value_shield_violation,
            )

            ep_shield_violation += safety_violation
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
                    last_value_shield_violation = torch.zeros(1)
                    
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, last_value_shield_violation, _ = agent.step(obs[idx].float())
                        if time_out:
                            _, last_value_r, last_value_c, last_value_shield_violation, _ = agent.step(
                                info['final_observation'][idx].float(),
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                        last_value_shield_violation = last_value_shield_violation.unsqueeze(0)
                    
                    if done or time_out:
                        self._log_metrics(logger, idx)
                        logger.store({'Safety/EpCostShield': ep_shield_violation[idx]})
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                        ep_shield_violation[idx] = 0.0
                    
                    shield.reset()
                    buffer.finish_path(last_value_r, last_value_c, last_value_shield_violation, idx)

    def _get_shielded_actions(
        self,
        obs_tensor_for_policy: torch.Tensor,
        info: Dict,
        agent: ConstraintActorQAndVCritic,
        shield: VectorizedShield,
        normalizer: Normalizer,
        shield_step_condition: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced shielded action selection with safety memory for vectorized environments."""
        # Check presafety condition
        unsafe_mask = shield._check_presafety_condition(info, enhanced_safety=0.15)
        unsafe_mask = torch.from_numpy(unsafe_mask).to(self._device)
        dones = torch.zeros(self.vector_env_nums, device=self._device).bool()

        if not shield_step_condition:
            agent_pos, agent_mat, goal_pos, hazards, pillars, gremlins = shield.process_info(info, self.vector_env_nums)
        
        # if self.warm_up_condition and unsafe_mask.any() and shield_step_condition:
        if shield.prediction_horizon > 0 and unsafe_mask.any() and shield_step_condition:
            agent_pos, agent_mat, goal_pos, hazards, pillars, gremlins = shield.process_info(info, self.vector_env_nums)
            dp_input = t2numpy(shield.dp_input)
            shield.update_robot_actual_history(agent_pos)

            dp_acp_region = shield.robot_conformal_threshold
            dp_acp_region = min(dp_acp_region, 0.1)

            iter_nbr = 0
            max_iter_nbr = 1

            while not dones.all() and iter_nbr < max_iter_nbr:
                acts, value_r, value_c, value_shield_violation, logps = agent.step_with_multiple_samples(obs_tensor_for_policy, n_samples=shield.sampling_nbr)
                action_clipped = np.clip(t2numpy(acts), self.action_space.low, self.action_space.high)
                
                is_safe, min_indices, safety_measure = shield.sample_safe_actions(
                    dp_input,
                    agent_pos,
                    agent_mat,
                    goal_pos,
                    hazards,
                    pillars,
                    gremlins,
                    first_action=action_clipped,
                    policy=agent.step,
                    dp_acp_region=dp_acp_region,
                    device=self._device,
                    selection_method='top-k',
                    k=max(shield.sampling_nbr // 5, 1),
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
                logp = logps[min_indices, np.arange(len(min_indices))]
                if iter_nbr == 0:
                    final_actions = act.clone()
                    final_logps = logp.clone()
                else:
                    final_actions[dones] = act[dones]
                    final_logps[dones] = logp[dones]
                iter_nbr += 1
            shield.update_conformality_scores()
            shield._set_conformal_thresholds()
        else:
            act, value_r, value_c, value_shield_violation, logp = agent.step(obs_tensor_for_policy)
            shield.shield_triggered = False

        shield.prev_action = act
        return act, value_r, value_c, value_shield_violation, logp