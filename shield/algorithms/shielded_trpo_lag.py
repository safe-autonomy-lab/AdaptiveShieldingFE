import torch

from omnisafe.algorithms import registry
from shield.algorithms.shielded_trpo import ShieldedTRPO
from omnisafe.common.lagrange import Lagrange
import numpy as np


@registry.register
class ShieldedTRPOLag(ShieldedTRPO):
    def _init(self) -> None:
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
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