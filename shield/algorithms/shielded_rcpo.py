from __future__ import annotations

import time
import torch
import numpy as np

from omnisafe.algorithms import registry
from omnisafe.common.lagrange import Lagrange

from shield.onpolicy_wrapper import ShieldedNaturalPG


@registry.register
class ShieldedRCPO(ShieldedNaturalPG):
    def _init(self) -> None:
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/AcceptanceStep')
        self._logger.register_key('Safety/ShieldTriggeredCount')
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

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
    
    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        self._lagrange.update_lagrange_multiplier(Jc)
        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor, safe_prob: torch.Tensor) -> torch.Tensor:
        safety_bonus = self._shield.safety_bonus
        shield_penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r + safety_bonus * safe_prob - shield_penalty * adv_c) / (1.0 + safety_bonus + shield_penalty)
    
        # below, is a legacy form of SRO, I have tested with other option for SRO, but many of them is kind of conservative..
        # elif self._cfgs.shield_cfgs['penalty_type'] == 'mul':
        #     shield_penalty = self._lagrange.lagrangian_multiplier.item()
        #     return (adv_r * safe_prob - shield_penalty * adv_c) / (1.0 + shield_penalty)
        # elif self._cfgs.shield_cfgs['penalty_type'] == 'shield':
        #     shield_penalty = self._lagrange.lagrangian_multiplier.item()
        #     return (adv_r - shield_penalty * adv_c) / (1.0 + shield_penalty)

    def _loss_pi(self, obs, act, logp, adv):
        return super()._loss_pi(obs, act, logp, adv)