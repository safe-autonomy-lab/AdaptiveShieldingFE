"""Implementation of Wrapper for On-Poilcy Algorithms for Shield"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient

from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)

from omnisafe.common.logger import Logger
from omnisafe.common.lagrange import Lagrange
from shield.model.constraint_actor_q_and_v_critic import ConstraintActorQAndVCritic
from shield.vectorized_shield import VectorizedShield
from shield.adapter_wrapper import ShieldedOnPolicyAdapter
from shield.util import load_model


@registry.register
class ShieldedPolicyGradient(PolicyGradient):
    def _init_env(self) -> None:
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
    
    def _init_model(self) -> None:
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
        
        env_id = self._env_id
        env_info = env_id.split('-')[0]
        shield_config = self._cfgs.shield_cfgs
        shield_config['env_info'] = env_info
        shield_config['vector_env_nums'] = self.vector_env_nums
        self.env_info = env_info
        dynamics_model_type = shield_config['dynamics_model']
        prediction_horizon = shield_config.get('prediction_horizon')
        
        is_swimmer = 'Swimmer' in env_info
        is_cheetah = 'Cheetah' in env_info
            
        dynamics_predictor = load_model(
            f'saved_files/dynamics_predictor/{env_info}-v1',
            dynamics_model_type,
            prediction_horizon=prediction_horizon,
            seed=self._cfgs.seed,
        )
        mo_predictor = None
        self._shield = VectorizedShield(dynamic_predictor=dynamics_predictor, mo_predictor=mo_predictor, dynamics_model_type=dynamics_model_type, **shield_config)
        self._shield.is_swimmer = is_swimmer
        self._shield.is_cheetah = is_cheetah
        # reward: shielding + sro, shield: only shielding, sro: only sro
        self.use_sro = shield_config['penalty_type'] in ['reward', 'sro']
        self.safety_bonus = self._shield.safety_bonus
        self._obs_normalizer = None
        if self._cfgs.algo_cfgs.obs_normalize:
            self._obs_normalizer = self._env._env.get_obs_normalizer()

    def _init_log(self):
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        what_to_save['v_reward'] = self._actor_critic.reward_critic
        what_to_save['v_cost'] = self._actor_critic.cost_v_critic
        what_to_save['q_cost'] = self._actor_critic.cost_q_critic
        what_to_save['reward_optimizers'] = self._actor_critic.reward_critic_optimizer
        what_to_save['cost_v_optimizers'] = self._actor_critic.cost_v_critic_optimizer
        what_to_save['cost_q_optimizers'] = self._actor_critic.cost_q_critic_optimizer
        what_to_save['scheduler'] = self._actor_critic.actor_scheduler if hasattr(self._actor_critic, 'actor_scheduler') else None
        
        if hasattr(self, '_lagrange') and self._lagrange is not None:
            what_to_save['lagrange'] = self._lagrange
        
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._what_to_save = what_to_save
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            'Metrics/EpRet',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpCost',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio', min_and_max=True)
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.register_key('Train/PolicyStd')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self.logger.register_key(env_spec_key)
        
        self._logger.register_key('Safety/EpCostShield')
        self._logger.register_key('Safety/ShieldViolation')

    def _update(self):
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic_q_and_v_critic(obs, act, target_value_c, adv_c)
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': final_kl,
            },
        )

    def _update_cost_critic_q_and_v_critic(self, obs: torch.Tensor, act: torch.Tensor, target_value_c: torch.Tensor, target_value_adv_c: torch.Tensor) -> None:
        self._actor_critic.cost_q_critic_optimizer.zero_grad()
        self._actor_critic.cost_v_critic_optimizer.zero_grad()

        cost_v_grad = self._actor_critic.cost_v_critic(obs)[0]
        loss_v = nn.functional.mse_loss(cost_v_grad, target_value_c)
        cost_v = cost_v_grad.detach()
        loss_adv = nn.functional.mse_loss(self._actor_critic.cost_q_critic(obs, act)[0] - cost_v, target_value_adv_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_q_critic.parameters():
                loss_v += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef
            for param in self._actor_critic.cost_v_critic.parameters():
                loss_adv += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss_v.backward()
        loss_adv.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_q_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
            clip_grad_norm_(
                self._actor_critic.cost_v_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
            
        distributed.avg_grads(self._actor_critic.cost_q_critic)
        distributed.avg_grads(self._actor_critic.cost_v_critic)
        
        self._actor_critic.cost_q_critic_optimizer.step()
        self._actor_critic.cost_v_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss_v.mean().item() + loss_adv.mean().item()})
    
    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:

        safe_prob = self._compute_safe_prob(obs, act, logp) if self.use_sro else 0
        adv = self._compute_adv_surrogate(adv_r, adv_c, safe_prob)
        loss = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()
    
    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        safe_prob: torch.Tensor,
    ) -> torch.Tensor:
        return (adv_r + self.safety_bonus * safe_prob) / (1. + self.safety_bonus)
    
    def _loss_pi(self, obs, act, logp, adv):
        return super()._loss_pi(obs, act, logp, adv)

    # This is where SRO works !!
    def _compute_safe_prob(self, obs, act, logp, num_samples=50, std_dev=1e-1):
        with torch.no_grad():
            # We approximate the probability integral over epsilon ball around a
            act_expanded = act.unsqueeze(1).expand(-1, num_samples, -1)
            obs_expanded = obs.unsqueeze(1).expand(-1, num_samples, -1)
            noise = torch.randn_like(act_expanded) * std_dev
            sampled_acts = act_expanded + noise
            sampled_acts_flat = sampled_acts.reshape(-1, act.shape[1])
            obs_flat = obs_expanded.reshape(-1, obs.shape[1])

            value_q_flat = self._actor_critic.forward_cost_q_critic(obs_flat, sampled_acts_flat).reshape(-1, num_samples)
            approximated_q = value_q_flat.mean(dim=1)

            value_cost = self._actor_critic.cost_v_critic(obs)[0]
            safe_denominator = torch.clamp(value_cost, min=1e-1)
            safe_prob = - torch.clamp(approximated_q * logp.exp() * 2e-1 / safe_denominator, min=-1.0, max=0.0)
        return safe_prob


@registry.register
class ShieldedNaturalPG(ShieldedPolicyGradient):
    _fvp_obs: torch.Tensor

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/Alpha')
        self._logger.register_key('Misc/FinalStepNorm')
        self._logger.register_key('Misc/gradient_norm')
        self._logger.register_key('Misc/xHx')
        self._logger.register_key('Misc/H_inv_g')

    def _fvp(self, params: torch.Tensor) -> torch.Tensor:
        self._actor_critic.actor.zero_grad()
        q_dist = self._actor_critic.actor(self._fvp_obs)
        with torch.no_grad():
            p_dist = self._actor_critic.actor(self._fvp_obs)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        grads = torch.autograd.grad(
            kl,
            tuple(self._actor_critic.actor.parameters()),
            create_graph=True,
        )
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * params).sum()
        grads = torch.autograd.grad(
            kl_p,
            tuple(self._actor_critic.actor.parameters()),
            retain_graph=False,
        )

        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
        distributed.avg_tensor(flat_grad_grad_kl)

        self._logger.store(
            {
                'Train/KL': kl.item(),
            },
        )
        return flat_grad_grad_kl + params * self._cfgs.algo_cfgs.cg_damping
    
    def _update_actor(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        safe_prob = self._compute_safe_prob(obs, act, logp)
        adv = self._compute_adv_surrogate(adv_r, adv_c, safe_prob)
        loss = self._loss_pi(obs, act, logp, adv)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

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
            },
        )

    def _update(self) -> None:
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )
        self._update_actor(obs, act, logp, adv_r, adv_c)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, target_value_r, target_value_c, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                act,
                target_value_r,
                target_value_c,
                adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic_q_and_v_critic(obs, act, target_value_c, adv_c)

        self._logger.store(
            {
                'Train/StopIter': self._cfgs.algo_cfgs.update_iters,
                'Value/Adv': adv_r.mean().item(),
            },
        )
