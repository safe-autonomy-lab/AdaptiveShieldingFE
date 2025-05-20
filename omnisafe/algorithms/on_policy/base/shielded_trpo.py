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
"""Implementation of the TRPO algorithm."""

from __future__ import annotations

import time
import torch
from torch.distributions import Distribution

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.shielded_natural_pg import ShieldedNaturalPG
from omnisafe.utils import distributed
from omnisafe.adapter.shielded_onpolicy_adapter import ShieldedOnPolicyAdapter
from omnisafe.models.actor_critic.constraint_actor_q_and_v_critic import ConstraintActorQAndVCritic
from omnisafe.shield.vectorized_shield import VectorizedShield
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class ShieldedTRPO(ShieldedNaturalPG):
    """The Trust Region Policy Optimization (TRPO) algorithm.

    References:
        - Title: Trust Region Policy Optimization
        - Authors: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel.
        - URL: `TRPO <https://arxiv.org/abs/1502.05477>`_
    """
    def _init_log(self) -> None:
        """Log the Trust Region Policy Optimization specific information.

        +---------------------+-----------------------------+
        | Things to log       | Description                 |
        +=====================+=============================+
        | Misc/AcceptanceStep | The acceptance step size.   |
        +---------------------+-----------------------------+
        """
        super()._init_log()
        self._logger.register_key('Misc/AcceptanceStep')

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.
        """
        self.vector_env_nums = self._cfgs.train_cfgs.vector_env_nums
        self._env: ShieldedOnPolicyAdapter = ShieldedOnPolicyAdapter(
            self._env_id,
            self.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self.vector_env_nums
        )
        self._obs_normalizer = None

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: ConstraintActorQAndVCritic = ConstraintActorQAndVCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )
        self._cfgs.shield_cfgs['env_info'] = self._cfgs.env_id.split('-')[0]
        self._cfgs.shield_cfgs['vector_env_nums'] = self.vector_env_nums
        self._shield = VectorizedShield(**self._cfgs.shield_cfgs)
        self._obs_normalizer = None
        if self._cfgs.algo_cfgs.obs_normalize:
            self._obs_normalizer = self._env._env.get_obs_normalizer()
        
    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()

            rollout_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
                shield=self._shield,
                normalizer=self._obs_normalizer,
            )
            
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    # pylint: disable-next=too-many-arguments,too-many-locals,arguments-differ
    def _search_step_size(
        self,
        step_direction: torch.Tensor,
        grads: torch.Tensor,
        p_dist: Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
        loss_before: torch.Tensor,
        total_steps: int = 15,
        decay: float = 0.8,
    ) -> tuple[torch.Tensor, int]:
        """TRPO performs `line-search <https://en.wikipedia.org/wiki/Line_search>`_ until constraint satisfaction.

        .. hint::
            TRPO search around for a satisfied step of policy update to improve loss and reward performance. The search
            is done by line-search, which is a way to find a step size that satisfies the constraint. The constraint is
            the KL-divergence between the old policy and the new policy.

        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            logp (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            adv_c (torch.Tensor): The cost advantage.
            loss_pi_before (float): The loss of the policy before the update.
            total_steps (int, optional): The total steps to search. Defaults to 15.
            decay (float, optional): The decay rate of the step size. Defaults to 0.8.

        Returns:
            The tuple of final update direction and acceptance step size.
        """
        # How far to go in a single update
        step_frac = 1.0
        # Get old parameterized policy expression
        theta_old = get_flat_params_from(self._actor_critic.actor)
        # Change expected objective function gradient = expected_imrpove best this moment
        expected_improve = grads.dot(step_direction)
        final_kl = 0.0

        # While not within_trust_region and not out of total_steps:
        for step in range(total_steps):
            # update theta params
            new_theta = theta_old + step_frac * step_direction
            # set new params as params of net
            set_param_values_to_model(self._actor_critic.actor, new_theta)

            with torch.no_grad():
                loss = self._loss_pi(obs, act, logp, adv)
                # compute KL distance between new and old policy
                q_dist = self._actor_critic.actor(obs)
                # KL-distance of old p-dist and new q-dist, applied in KLEarlyStopping
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()
                kl = distributed.dist_avg(kl).mean().item()
            # real loss improve: old policy loss - new policy loss
            loss_improve = loss_before - loss
            # average processes.... multi-processing style like: mpi_tools.mpi_avg(xxx)
            loss_improve = distributed.dist_avg(loss_improve)
            self._logger.log(
                f'Expected Improvement: {expected_improve} Actual: {loss_improve.item()}',
            )
            if not torch.isfinite(loss):
                self._logger.log('WARNING: loss_pi not finite')
            elif loss_improve.item() < 0:
                self._logger.log('INFO: did not improve improve <0')
            elif kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                acceptance_step = step + 1
                self._logger.log(f'Accept step at i={acceptance_step}')
                final_kl = kl
                break
            step_frac *= decay
        else:
            self._logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        set_param_values_to_model(self._actor_critic.actor, theta_old)

        self._logger.store(
            {
                'Train/KL': final_kl,
            },
        )

        return step_frac * step_direction, acceptance_step

    def _update_actor(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        shield_adv_c: torch.Tensor,
    ) -> None:
        """Update policy network.

        Trust Policy Region Optimization updates policy network using the
        `conjugate gradient <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_ algorithm,
        following the steps:

        - Compute the gradient of the policy.
        - Compute the step direction.
        - Search for a step size that satisfies the constraint.
        - Update the policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
            shield_violation (torch.Tensor): The shield violation tensor.
            add_shield_penalty (bool, optional): Whether to add shield penalty. Defaults to False.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        _, cost_q = self._actor_critic.get_cost_q_critic_action_gradient(obs, act)
        
        with torch.no_grad():
            value_cost = self._actor_critic.cost_v_critic(obs)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            # Ensure value_cost is positive and non-zero
            safe_denominator = torch.clamp(value_cost[0], min=epsilon)
            # Calculate safe probability and clamp between 0 and 1
            safe_prob = 1. - torch.clamp(cost_q * logp.exp() / safe_denominator, min=0.0, max=1.0).detach()

        # adv = self._compute_adv_surrogate(adv_r , adv_c, shield_adv_c)
        adv = self._compute_adv_surrogate(adv_r, safe_prob, shield_adv_c)
        
        loss = self._loss_pi(obs, act, logp, adv)
        
        loss_before = distributed.dist_avg(loss)
        p_dist = self._actor_critic.actor(obs)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), f'x is not finite {x}'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, f'xHx is negative {xHx}'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), f'step_direction is not finite {step_direction}'

        step_direction, accept_step = self._search_step_size(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv=adv,
            loss_before=loss_before,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss = self._loss_pi(obs, act, logp, adv)

        self._logger.store(
            {
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
                'Misc/AcceptanceStep': accept_step,
            },
        )
