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
"""Implementation of the Lagrange version of the TRPO algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.shielded_trpo import ShieldedTRPO
from omnisafe.common.lagrange import Lagrange
import numpy as np


@registry.register
class ShieldedTRPOLag(ShieldedTRPO):
    """The Lagrange version of the TRPO algorithm.

    A simple combination of the Lagrange method and the Trust Region Policy Optimization algorithm.
    """

    def _init(self) -> None:
        """Initialize the TRPOLag specific model.

        The TRPOLag algorithm uses a Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        self._shield_lagrange: Lagrange = Lagrange(cost_limit = 0., lagrangian_multiplier_init=0.01, lambda_lr=0.035, lambda_optimizer='Adam', lagrangian_upper_bound=5.)

    def _init_log(self) -> None:
        """Log the TRPOLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)
        self._logger.register_key('Safety/ShieldLagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old}(a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        Jc_shield = self._logger.get_stats('Safety/EpCostShield')[0]
        assert not np.isnan(Jc_shield), 'shield cost for updating lagrange multiplier is nan'
        self._shield_lagrange.update_lagrange_multiplier(Jc_shield)
        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})
        self._logger.store({'Safety/ShieldLagrangeMultiplier': self._shield_lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, safe_prob: torch.Tensor, adv_c_shield: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        TRPOLag uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            safe_prob (torch.Tensor): The ``safe_probability`` sampled from buffer.
            adv_c_shield (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        safety_bonus = torch.abs(self._shield.safety_bonus)
        if self._cfgs.shield_cfgs['penalty_type'] == 'mul':
            shield_penalty = self._shield_lagrange.lagrangian_multiplier.item()
            return (adv_r * safe_prob - shield_penalty * adv_c_shield) / (1.0 + shield_penalty)
        elif self._cfgs.shield_cfgs['penalty_type'] == 'reward':
            shield_penalty = self._shield_lagrange.lagrangian_multiplier.item()
            return (adv_r + safety_bonus * safe_prob - shield_penalty * adv_c_shield) / (1.0 + safety_bonus + shield_penalty)
        # This is shield only case without using safety bonus
        elif self._cfgs.shield_cfgs['penalty_type'] == 'shield':
            shield_penalty = self._shield_lagrange.lagrangian_multiplier.item()
            return (adv_r - shield_penalty * adv_c_shield) / (1.0 + shield_penalty)
        else:
            raise ValueError(f'Invalid penalty type: {self._cfgs.shield_cfgs["penalty_type"]}') 