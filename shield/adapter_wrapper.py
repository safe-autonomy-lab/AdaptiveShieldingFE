"""Shielded OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

from tkinter import SEL_FIRST
from typing import Tuple, Dict
import os

import torch
from rich.progress import track

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.common.normalizer import Normalizer
from omnisafe.utils.config import Config
from omnisafe.utils import distributed

from shield.model.constraint_actor_q_and_v_critic import ConstraintActorQAndVCritic
from shield.vectorized_shield import VectorizedShield
import numpy as np
import time


class ShieldedOnPolicyAdapter(OnPolicyAdapter):
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
        self._obs_normalizer = None
        self.vector_env_nums = num_envs
        self.env_id = env_id
        self.is_circle = 'Circle' in self.env_id
        self.is_velocity = 'Velocity' in self.env_id
        self._reset_log()

    def _get_agent_pos_mat(self, info: Dict, shield: VectorizedShield):
        if self.is_velocity:
            # For velocity environments, we track velocity for dynamics prediction
            return info['x_velocity'].reshape(-1, 1), None
        return shield._process_agent_information(info)

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorQAndVCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
        shield: VectorizedShield,
        normalizer: Normalizer = None,
    ) -> None:
        self._reset_log()
        obs, info = self.reset()
        # Use wrapper-provided hidden dim when available; info['hidden_parameters_dim']
        # reflects the underlying env and can differ when FE basis is used.
        self.hidden_parameters_dim = int(getattr(self._env, "hidden_parameters_dim", info['hidden_parameters_dim'][0]))
        if hasattr(self._env, 'get_slices'):
            robot_slices = self._env.get_slices()['robot']
        else:
            # velocity environments have a different way of getting the robot slices
            robot_slices = info['robot'][0]
        
        use_oracle = shield.dynamics_model_type in ['oracle', 'pem']
        action_low = torch.from_numpy(self.action_space.low).to(self._device).float()
        action_high = torch.from_numpy(self.action_space.high).to(self._device).float()
        
        # Initialize shield properties
        shield.is_circle = self.is_circle
        shield.is_velocity = self.is_velocity
        shield.robot_slices = robot_slices
        shield.range_limit = info['sigwalls_loc'][0] if self.is_circle else None
        if shield.prediction_horizon is None or shield.prediction_horizon <= 0:
            out_size = getattr(shield.dynamic_predictor, "output_size", None)
            if out_size is not None and len(out_size) > 0:
                per_step_dim = 1 if shield.is_velocity else 2
                inferred = max(1, int(out_size[0]) // per_step_dim)
                shield.prediction_horizon = inferred
        
        # Initial weights
        weights = torch.zeros(self.vector_env_nums, shield.n_basis).to(self._device)
        shield.coeffs_for_dynamics_prediction = weights
        shield.normalized_coeffs_for_dynamics_prediction = weights
        # If we use sro, we do not trigger shield at all, reward: shielding + sro, shield: only shielding
        self.trigger_shield = shield.penalty_type in ["reward", "shield"]

        episode_step = 0
        self.shield_triggered_count = 0

        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            agent_pos, agent_mat = self._get_agent_pos_mat(info, shield)
            
            # Update weights if needed (online adaptation)
            shield_trigger = False
            if shield.prev_dp_input is not None and not use_oracle:
                shield_trigger = shield.update_weights(episode_step)
                shield_trigger = shield_trigger and logger.current_epoch > shield.warm_up_epochs
            
            # Prepare observation for policy and dynamics input
            original_robot_obs = info['original_obs'][:, robot_slices] if 'original_obs' in info else obs[:, robot_slices]
            if not use_oracle:
                obs[:, -self.hidden_parameters_dim:] = shield.normalized_coeffs_for_dynamics_prediction
            else:
                hidden_parameters = info['original_obs'][:, -self.hidden_parameters_dim:]
                original_robot_obs = torch.cat([original_robot_obs, hidden_parameters], dim=-1)

            shield.prepare_dp_input(original_robot_obs, agent_pos, agent_mat, device=self._device)
            # Get action (shielded or vanilla)
            act, value_r, value_c, logp = self._get_shielded_actions(
                obs, info, agent, shield, action_low, action_high, shield_trigger
            )
            
            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            
            # Logging and metrics
            unsafe_condition = torch.from_numpy(shield._check_presafety_condition(info, enhanced_safety=0.)).to(self._device)
            safety_violation = torch.logical_or(unsafe_condition, cost.to(self._device)).float()
            
            self._log_value(reward=reward, cost=cost, info=info)
            logger.store({
                'Safety/ShieldViolation': safety_violation.cpu().mean(),
                'Value/reward': value_r,
                'Safety/ShieldTriggeredCount': self.shield_triggered_count / (episode_step + 1)
            })
            
            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})

            buffer.store(
                obs=obs, act=act, reward=reward, cost=cost,
                value_r=value_r, value_c=value_c, logp=logp,
            )

            obs = next_obs

            episode_step += 1
            epoch_end = step >= steps_per_epoch - 1

            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(f'\nWarning: trajectory cut off in {self._env.num_envs - num_dones} environments.')

            # Handle terminations
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r, last_value_c = torch.zeros(1), torch.zeros(1)
                    
                    if not done:
                        temp_obs = obs[idx] if epoch_end else info['final_observation'][idx]
                        _, last_value_r, last_value_c, _ = agent.step(temp_obs.float())
                        last_value_r, last_value_c = last_value_r.unsqueeze(0), last_value_c.unsqueeze(0)
                    
                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)
                        self._ep_ret[idx], self._ep_cost[idx], self._ep_len[idx] = 0.0, 0.0, 0.0
                        self.shield_triggered_count = 0
                        episode_step = 0
                        shield.reset()

                    buffer.finish_path(last_value_r, last_value_c, idx)
        
        self._save_shield_normalizer(shield, logger)

    def _save_shield_normalizer(self, shield: VectorizedShield, logger: Logger) -> None:
        if distributed.get_rank() != 0:
            return
        if not hasattr(shield, "normalizer") or shield.normalizer is None:
            return
        use_fe = bool(getattr(self._cfgs.shield_cfgs, "use_fe_representation", False))
        folder_name = "fe" if use_fe else "oracle"
        env_id = getattr(self, "env_id", self._env_id)
        prediction_horizon = getattr(self._cfgs.shield_cfgs, "prediction_horizon", None)
        algo = getattr(self._cfgs, "algo", "algo")
        penalty_type = str(getattr(self._cfgs.shield_cfgs, "penalty_type", "")).lower()
        if penalty_type == "reward":
            algo = f"{algo}withSRO"
        elif penalty_type == "sro":
            base_algo = algo
            if base_algo.startswith("Shielded"):
                base_algo = base_algo[len("Shielded") :]
            algo = f"{base_algo}withSRO"
        seed = int(getattr(self._cfgs, "seed", 0))
        results_dir = os.path.join(
            "results",
            folder_name,
            env_id,
            f"h{prediction_horizon}",
            algo,
            f"seed{seed}",
        )
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"shield_normalizer.pt")
        payload = {
            "state_dict": shield.normalizer.state_dict(),
            "n_basis": getattr(shield, "n_basis", None),
            "prediction_horizon": prediction_horizon,
            "env_id": env_id,
            "algo": algo,
            "seed": seed,
        }
        torch.save(payload, save_path)

    def _get_shielded_actions(
        self,
        obs_tensor_for_policy: torch.Tensor,
        info: Dict,
        agent: ConstraintActorQAndVCritic,
        shield: VectorizedShield,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        shield_trigger: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced shielded action selection with safety memory for vectorized environments."""
        # Check presafety condition
        unsafe_mask = torch.from_numpy(shield._check_presafety_condition(info, enhanced_safety=0.15)).to(self._device)
        
        # Update conformal prediction history
        shield.update_robot_actual_history(shield.dp_y)
        shield.step_last_triggered += 1
        
        if shield.shield_triggered:
            shield.update_conformality_scores()
            shield._set_conformal_thresholds()
            shield.shield_triggered = False

        # Shielding decision logic
        should_shield = (
            self.trigger_shield and 
            shield.step_last_triggered > shield.idle_condition and 
            shield_trigger and 
            shield.prediction_horizon > 0 and 
            unsafe_mask.any()
        )
        # if True: # this line is for debugging, triggering shield always
        if should_shield:
            agent_pos, _ = self._get_agent_pos_mat(info, shield)
            if not self.is_velocity:
                _, _, buttons, goals, hazards, vases, pillars, push_boxes, gremlins, circle = shield.process_info(info, self.vector_env_nums)
            else:
                buttons, goals, hazards, vases, pillars, push_boxes, gremlins, circle = [None] * 8
                
            # scale controlling diversity of actions sampled from policy
            acts, value_r, value_c, logps = agent.sample(obs_tensor_for_policy, n_samples=shield.sampling_nbr, scale=shield.scale)
            action_clipped = acts.clamp(action_low, action_high)
            
            _, min_indices, _ = shield.sample_safe_actions(
                shield.dp_input, agent_pos, buttons, goals, hazards, vases, pillars, 
                push_boxes, gremlins, circle, first_action=action_clipped,
                device=self._device, selection_method='top-k',
                k=max(shield.sampling_nbr // 5, 1),
            )
            
            act = acts[min_indices, np.arange(len(min_indices))]
            logp = logps[min_indices, np.arange(len(min_indices))]
            
            self.shield_triggered_count += 1
            shield.shield_triggered = True
            shield.step_last_triggered = 0
        else:
            act, value_r, value_c, logp = agent.step(obs_tensor_for_policy)
            shield.shield_triggered = False

        shield.prev_action = act
        return act, value_r, value_c, logp
