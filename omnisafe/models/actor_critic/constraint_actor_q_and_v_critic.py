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
"""Implementation of ConstraintActorCritic."""

from __future__ import annotations

import torch
from torch import optim

from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class ConstraintActorQAndVCritic(ActorCritic):
    """ConstraintActorCritic is a wrapper around ActorCritic that adds a cost critic to the model.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+-----------------------------------------------+
    | Model           | Description                                   |
    +=================+===============================================+
    | Actor           | Input is observation. Output is action.       |
    +-----------------+-----------------------------------------------+
    | Reward V Critic | Input is observation. Output is reward value. |
    +-----------------+-----------------------------------------------+
    | Cost V Critic   | Input is observation. Output is cost value.   |
    +-----------------+-----------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        cost_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)
        self.cost_v_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic('v')
        self.add_module('cost_v_critic', self.cost_v_critic)

        self.cost_q_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic('q')
        self.add_module('cost_q_critic', self.cost_q_critic)

        self.shield_violation_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic('v')
        self.add_module('shield_violation_critic', self.shield_violation_critic)

        if model_cfgs.critic.lr is not None:
            self.cost_v_critic_optimizer: optim.Optimizer
            self.cost_v_critic_optimizer = optim.Adam(
                self.cost_v_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

            self.shield_violation_critic_optimizer: optim.Optimizer
            self.shield_violation_critic_optimizer = optim.Adam(
                self.shield_violation_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

            self.cost_q_critic_optimizer: optim.Optimizer
            self.cost_q_critic_optimizer = optim.Adam(
                self.cost_q_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

    def step(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            value_c = self.cost_v_critic(obs)
            shield_violation = self.shield_violation_critic(obs)
            action = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)

        return action, value_r[0], value_c[0], shield_violation[0], log_prob
    
    def gradient_adjusted_step(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            action = self.actor.predict(obs, deterministic=False)
            # log_prob = self.actor.log_prob(action)
        
        gradient, cost_q_value = self.get_cost_q_critic_action_gradient(obs, action)
        action = action - gradient
        with torch.no_grad():
            log_prob = self.actor.log_prob(action)
            value_r = self.reward_critic(obs)
            value_c = self.cost_v_critic(obs)
            value_shield_violation = self.shield_violation_critic(obs)

        return action, value_r[0], value_c[0], value_shield_violation[0], log_prob, gradient, cost_q_value

    def step_with_multiple_samples(self, obs: torch.Tensor, n_samples: int = 10) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            action = self.actor.predict_with_multiple_samples(obs, n_samples)
            log_prob = self.actor.log_prob(action)
            value_r = self.reward_critic(obs)
            value_c = self.cost_v_critic(obs)
            value_shield_violation = self.shield_violation_critic(obs)
        return action, value_r[0], value_c[0], value_shield_violation[0], log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.actor.predict(obs, deterministic=True)
            log_prob = self.actor.log_prob(action)
        return log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, deterministic=deterministic)

    def get_cost_q_critic_action_gradient(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient of cost critic with respect to action input.

        Args:
            obs (torch.Tensor): Observation tensor
            action (torch.Tensor): Action tensor to compute gradient for

        Returns:
            torch.Tensor: Gradient of cost critic with respect to action
        """
        # Create action tensor that requires gradient
        action_grad = action.detach().clone()
        action_grad.requires_grad_(True)

        # Temporarily disable gradient computation for cost critic parameters
        for param in self.cost_q_critic.parameters():
            param.requires_grad_(False)

        # Forward pass through cost critic using action_grad instead of action
        cost_q_value = self.cost_q_critic(obs, action_grad)
        cost_q_value = cost_q_value[0]  # Get first element of the list
        # Make sure cost_value is properly shaped for gradient computation
        if len(cost_q_value.shape) > 1:
            cost_q_value = cost_q_value.mean(dim=0)  # Average across batch dimension if needed
    
        # Compute gradient
        gradient = torch.autograd.grad(
            outputs=cost_q_value,
            inputs=action_grad,
            grad_outputs=torch.ones_like(cost_q_value).to(cost_q_value.device),
            create_graph=False,
            retain_graph=False,
        )[0]
        
        # Re-enable gradient computation for cost critic parameters
        for param in self.cost_q_critic.parameters():
            param.requires_grad_(True)
    
        return gradient, cost_q_value