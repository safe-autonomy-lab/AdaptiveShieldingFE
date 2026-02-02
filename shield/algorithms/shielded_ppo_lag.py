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
"""Implementation of the Lagrange version of the PPO algorithm."""

import numpy as np
import torch

from omnisafe.algorithms import registry

from omnisafe.common.lagrange import Lagrange
from shield.algorithms.shielded_ppo import ShieldedPPO


@registry.register
class ShieldedPPOLag(ShieldedPPO):
    def _init(self) -> None:
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)
        self._logger.register_key('Safety/ShieldTriggeredCount')

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