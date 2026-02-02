from __future__ import annotations

import time
import torch
from torch.distributions import Distribution

from omnisafe.algorithms import registry

from omnisafe.utils import distributed

from omnisafe.utils.tools import (
    get_flat_params_from,
    set_param_values_to_model,
)

from shield.onpolicy_wrapper import ShieldedNaturalPG


@registry.register
class ShieldedTRPO(ShieldedNaturalPG):
    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/AcceptanceStep')
        self._logger.register_key('Safety/ShieldTriggeredCount')
    def learn(self) -> tuple[float, float, float]:
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
        step_frac = 1.0
        theta_old = get_flat_params_from(self._actor_critic.actor)
        expected_improve = grads.dot(step_direction)
        final_kl = 0.0

        for step in range(total_steps):
            new_theta = theta_old + step_frac * step_direction
            set_param_values_to_model(self._actor_critic.actor, new_theta)

            with torch.no_grad():
                loss = self._loss_pi(obs, act, logp, adv)
                q_dist = self._actor_critic.actor(obs)
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()
                kl = distributed.dist_avg(kl).mean().item()
            loss_improve = loss_before - loss
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
    
    def _loss_pi(self, obs, act, logp, adv):
        return super()._loss_pi(obs, act, logp, adv)