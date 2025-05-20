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
"""Wrapper for the environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.common import Normalizer
from omnisafe.envs.core import CMDP, Wrapper


class TimeLimit(Wrapper):
    """Time limit wrapper for the environment.

    .. warning::
        The time limit wrapper only supports single environment.

    Examples:
        >>> env = TimeLimit(env, time_limit=100)

    Args:
        env (CMDP): The environment to wrap.
        time_limit (int): The time limit for each episode.
        device (torch.device): The torch device to use.

    Attributes:
        _time_limit (int): The time limit for each episode.
        _time (int): The current time step.
    """

    def __init__(self, env: CMDP, time_limit: int, device: torch.device) -> None:
        """Initialize an instance of :class:`TimeLimit`."""
        super().__init__(env=env, device=device)
        assert self.num_envs == 1, 'TimeLimit only supports single environment'
        self._time: int = 0
        self._time_limit: int = time_limit

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        .. note::
            Additionally, the time step will be reset to 0.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        self._time = 0
        return super().reset(seed=seed, options=options)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            Additionally, the time step will be increased by 1.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)

        self._time += 1
        truncated = torch.tensor(
            self._time >= self._time_limit,
            dtype=torch.bool,
            device=self._device,
        )

        return obs, reward, cost, terminated, truncated, info


class AutoReset(Wrapper):
    """Auto reset the environment when the episode is terminated.

    Examples:
        >>> env = AutoReset(env)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The torch device to use.
    """

    def __init__(self, env: CMDP, device: torch.device) -> None:
        """Initialize an instance of :class:`AutoReset`."""
        super().__init__(env=env, device=device)

        assert self.num_envs == 1, 'AutoReset only supports single environment'

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            If the episode is terminated, the environment will be reset. The ``obs`` will be the
            first observation of the new episode. And the true final observation will be stored in
            ``info['final_observation']``.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            new_obs, new_info = self.reset()
            assert (
                'final_observation' not in new_info
            ), 'info dict cannot contain key "final_observation" '
            assert 'final_info' not in new_info, 'info dict cannot contain key "final_info" '

            new_info['final_observation'] = obs
            new_info['final_info'] = info

            obs = new_obs
            info = new_info

        return obs, reward, cost, terminated, truncated, info


class ObsNormalize(Wrapper):
    """Normalize the observation.

    Examples:
        >>> env = ObsNormalize(env)
        >>> norm = Normalizer(env.observation_space.shape)  # load saved normalizer
        >>> env = ObsNormalize(env, norm)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The torch device to use.
        norm (Normalizer or None, optional): The normalizer to use. Defaults to None.
    """

    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        """Initialize an instance of :class:`ObsNormalize`."""
        super().__init__(env=env, device=device)
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box'
        self._obs_normalizer: Normalizer

        if norm is not None:
            self._obs_normalizer = norm.to(self._device)
        else:
            self._obs_normalizer = Normalizer(self.observation_space.shape, clip=5).to(self._device)

    def get_obs_normalizer(self) -> Normalizer:
        return self._obs_normalizer

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The observation and the ``info['final_observation']`` will be normalized.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)
        if 'final_observation' in info:
            final_obs_slice = info['_final_observation'] if self.num_envs > 1 else slice(None)
            info['final_observation'] = info['final_observation'].to(self._device)
            info['original_final_observation'] = info['final_observation']
            info['final_observation'][final_obs_slice] = self._obs_normalizer.normalize(
                info['final_observation'][final_obs_slice],
            )
        info['original_obs'] = obs
        obs = self._obs_normalizer.normalize(obs)
        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = super().reset(seed=seed, options=options)
        info['original_obs'] = obs
        obs = self._obs_normalizer.normalize(obs)
        return obs, info

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the observation normalizer.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize. When evaluating the saved model, the normalizer
            should be loaded.

        Returns:
            The saved components, that is the observation normalizer.
        """
        saved = super().save()
        saved['obs_normalizer'] = self._obs_normalizer
        return saved


class RewardNormalize(Wrapper):
    """Normalize the reward.

    Examples:
        >>> env = RewardNormalize(env)
        >>> norm = Normalizer(()) # load saved normalizer
        >>> env = RewardNormalize(env, norm)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The torch device to use.
        norm (Normalizer or None, optional): The normalizer to use. Defaults to None.
    """

    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        """Initialize an instance of :class:`RewardNormalize`."""
        super().__init__(env=env, device=device)
        self._reward_normalizer: Normalizer

        if norm is not None:
            self._reward_normalizer = norm.to(self._device)
        else:
            self._reward_normalizer = Normalizer((), clip=5).to(self._device)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The reward will be normalized for agent training. Then the original reward will be
            stored in ``info['original_reward']`` for logging.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['original_reward'] = reward
        reward = self._reward_normalizer.normalize(reward)
        return obs, reward, cost, terminated, truncated, info

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the reward normalizer.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize.

        Returns:
            The saved components, that is the reward normalizer.
        """
        saved = super().save()
        saved['reward_normalizer'] = self._reward_normalizer
        return saved


class CostNormalize(Wrapper):
    """Normalize the cost.

    Examples:
        >>> env = CostNormalize(env)
        >>> norm = Normalizer(()) # load saved normalizer
        >>> env = CostNormalize(env, norm)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The torch device to use.
        norm (Normalizer or None, optional): The normalizer to use. Defaults to None.
    """

    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        """Initialize an instance of :class:`CostNormalize`."""
        super().__init__(env=env, device=device)
        self._cost_normalizer: Normalizer

        if norm is not None:
            self._cost_normalizer = norm.to(self._device)
        else:
            self._cost_normalizer = Normalizer((), clip=5).to(self._device)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The cost will be normalized for agent training. Then the original reward will be stored
            in ``info['original_cost']`` for logging.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['original_cost'] = cost
        cost = self._cost_normalizer.normalize(cost)
        return obs, reward, cost, terminated, truncated, info

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the cost normalizer.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize.

        Returns:
            The saved components, that is the cost normalizer.
        """
        saved = super().save()
        saved['cost_normalizer'] = self._cost_normalizer
        return saved


class ActionScale(Wrapper):
    """Scale the action space to a given range.

    Examples:
        >>> env = ActionScale(env, low=-1, high=1)
        >>> env.action_space
        Box(-1.0, 1.0, (1,), float32)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The device to use.
        low (int or float): The lower bound of the action space.
        high (int or float): The upper bound of the action space.
    """

    def __init__(
        self,
        env: CMDP,
        device: torch.device,
        low: float,
        high: float,
    ) -> None:
        """Initialize an instance of :class:`ActionScale`."""
        super().__init__(env=env, device=device)
        assert isinstance(self.action_space, spaces.Box), 'Action space must be Box'

        self._old_min_action: torch.Tensor = torch.tensor(
            self.action_space.low,
            dtype=torch.float32,
            device=self._device,
        )
        self._old_max_action: torch.Tensor = torch.tensor(
            self.action_space.high,
            dtype=torch.float32,
            device=self._device,
        )

        min_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype) + low
        max_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype) + high
        self._action_space: spaces.Box = spaces.Box(
            low=min_action,
            high=max_action,
            shape=self.action_space.shape,
            dtype=self.action_space.dtype,  # type: ignore[arg-type]
        )

        self._min_action: torch.Tensor = torch.tensor(
            min_action,
            dtype=torch.float32,
            device=self._device,
        )
        self._max_action: torch.Tensor = torch.tensor(
            max_action,
            dtype=torch.float32,
            device=self._device,
        )

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The action will be scaled to the original range for agent training.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        action = self._old_min_action + (self._old_max_action - self._old_min_action) * (
            action - self._min_action
        ) / (self._max_action - self._min_action)

        return super().step(action)


class ActionRepeat(Wrapper):
    """Repeat action given times.

    Example:
        >>> env = ActionRepeat(env, times=3)
    """

    def __init__(
        self,
        env: CMDP,
        times: int,
        device: torch.device,
    ) -> None:
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            times: The number of times to repeat the action.
            device: The device to use.
        """
        super().__init__(env=env, device=device)
        self._times = times
        self._device = device

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run self._times timesteps of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        rewards, costs = torch.tensor(0.0).to(self._device), torch.tensor(0.0).to(self._device)
        for _step, _ in enumerate(range(self._times)):
            obs, reward, cost, terminated, truncated, info = super().step(action)
            rewards += reward
            costs += cost
            goal_met = info.get('goal_met', False)
            if terminated or truncated or goal_met:
                break
        info['num_step'] = _step + 1
        return obs, rewards, costs, terminated, truncated, info


class Unsqueeze(Wrapper):
    """Unsqueeze the observation, reward, cost, terminated, truncated and info.

    Examples:
        >>> env = Unsqueeze(env)
    """

    def __init__(self, env: CMDP, device: torch.device) -> None:
        """Initialize an instance of :class:`Unsqueeze`."""
        super().__init__(env=env, device=device)
        assert self.num_envs == 1, 'Unsqueeze only works with single environment'
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box'

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The vector information will be unsqueezed to (1, dim) for agent training.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        action = action.squeeze(0)
        obs, reward, cost, terminated, truncated, info = super().step(action)
        obs, reward, cost, terminated, truncated = (
            x.unsqueeze(0) for x in (obs, reward, cost, terminated, truncated)
        )
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns a new observation.

        .. note::
            The vector information will be unsqueezed to (1, dim) for agent training.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = super().reset(seed=seed, options=options)
        obs = obs.unsqueeze(0)
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, info



# Customized Dictionary Normalizer

import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

class Normalizer(torch.nn.Module):
    """A normalizer that tracks running mean and variance, supporting masked updates."""
    def __init__(self, shape: Tuple[int, ...], clip: float = 5.0):
        super().__init__()
        self.shape = shape
        self.clip = clip
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0.0))

    def update(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """Update running mean and variance with new data, optionally using a mask."""
        if mask is not None:
            # Expand mask to match x's shape for broadcasting
            mask = mask.bool()
            if mask.dim() < x.dim():
                mask = mask.unsqueeze(-1).expand_as(x)
            # Compute masked mean and variance
            masked_x = x * mask  # Zero out masked elements
            batch_count = mask.sum()
            if batch_count > 0:
                batch_mean = masked_x.sum(dim=0) / batch_count
                batch_var = ((masked_x - batch_mean * mask) ** 2).sum(dim=0) / batch_count
            else:
                return  # No valid data to update
        else:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)
            batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        if tot_count > 0:
            self.mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
            self.var = M2 / tot_count
            self.count = tot_count

    def normalize(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Normalize input tensor, optionally preserving masked values."""
        norm_x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        norm_x = torch.clamp(norm_x, -self.clip, self.clip)
        if mask is not None:
            # Preserve original values where mask is 0
            mask = mask.bool()
            if mask.dim() < norm_x.dim():
                mask = mask.unsqueeze(-1).expand_as(norm_x)
            norm_x = torch.where(mask, norm_x, x)
        return norm_x

    def to(self, device: torch.device) -> "Normalizer":
        """Move normalizer buffers to specified device."""
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
        return self

class DictObsNormalize(gym.Wrapper):
    """Normalize a dictionary-type observation space with MultiBinary masks.

    Examples:
        >>> env = ObsNormalize(env, device=torch.device('cuda'))
        >>> norms = {key: Normalizer(space.shape) for key, space in env.observation_space.spaces.items() if isinstance(space, spaces.Box)}  # Load saved normalizers
        >>> env = ObsNormalize(env, device=torch.device('cuda'), norm=norms)

    Args:
        env (gym.Env): The environment to wrap.
        device (torch.device): The torch device to use.
        norm (Dict[str, Normalizer] or None, optional): Pre-trained normalizers for Box spaces. Defaults to None.
    """

    def __init__(self, env: gym.Env, device: torch.device, norm: Optional[Dict[str, Normalizer]] = None) -> None:
        super().__init__(env)
        assert isinstance(self.observation_space, spaces.Dict), 'Observation space must be a Dict'

        self._device = device
        self._obs_normalizer: Dict[str, Normalizer] = {}

        # Define mask mappings for normalization
        self._mask_mappings = {
            'ped_pos': 'ped_mask',
            'veh_pos': 'veh_mask',
            'GT_pred_pos': 'GT_pred_mask' if 'GT_pred_mask' in self.observation_space.spaces else None,
        }

        if norm is not None:
            # Use provided normalizers for Box spaces
            for key, normalizer in norm.items():
                if key in self.observation_space.spaces and isinstance(self.observation_space.spaces[key], spaces.Box):
                    self._obs_normalizer[key] = normalizer.to(self._device)
        else:
            # Initialize normalizers only for Box spaces
            for key, space in self.observation_space.spaces.items():
                if isinstance(space, spaces.Box):
                    self._obs_normalizer[key] = Normalizer(space.shape, clip=5).to(self._device)

    def get_obs_normalizers(self) -> Dict[str, Normalizer]:
        """Return the dictionary of observation normalizers."""
        return self._obs_normalizer

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        # Move observation to device
        obs = {key: val.to(self._device) for key, val in obs.items()}

        # Update normalizers with masked data
        for key, val in obs.items():
            if key in self._obs_normalizer:
                mask_key = self._mask_mappings.get(key)
                mask = obs.get(mask_key) if mask_key in obs else None
                self._obs_normalizer[key].update(val, mask)

        # Normalize observation
        normalized_obs = {}
        for key, val in obs.items():
            if key in self._obs_normalizer:
                mask_key = self._mask_mappings.get(key)
                mask = obs.get(mask_key) if mask_key in obs else None
                normalized_obs[key] = self._obs_normalizer[key].normalize(val, mask)
            else:
                normalized_obs[key] = val  # Pass MultiBinary or other non-Box types through

        # Handle final observation if present
        if 'final_observation' in info:
            final_obs = info['final_observation']
            final_obs = {key: val.to(self._device) for key, val in final_obs.items()}
            info['original_final_observation'] = final_obs
            final_obs_slice = info['_final_observation'] if hasattr(self.env, 'num_envs') and self.env.num_envs > 1 else slice(None)
            normalized_final_obs = {}
            for key, val in final_obs.items():
                if key in self._obs_normalizer:
                    mask_key = self._mask_mappings.get(key)
                    mask = final_obs.get(mask_key) if mask_key in final_obs else None
                    norm_val = val.clone()
                    norm_val[final_obs_slice] = self._obs_normalizer[key].normalize(val[final_obs_slice], mask[final_obs_slice] if mask is not None else None)
                    normalized_final_obs[key] = norm_val
                else:
                    normalized_final_obs[key] = val
            info['final_observation'] = normalized_final_obs

        info['original_obs'] = obs
        return normalized_obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        # Move observation to device
        obs = {key: val.to(self._device) for key, val in obs.items()}

        # Update normalizers with masked data
        for key, val in obs.items():
            if key in self._obs_normalizer:
                mask_key = self._mask_mappings.get(key)
                mask = obs.get(mask_key) if mask_key in obs else None
                self._obs_normalizer[key].update(val, mask)

        # Normalize observation
        normalized_obs = {}
        for key, val in obs.items():
            if key in self._obs_normalizer:
                mask_key = self._mask_mappings.get(key)
                mask = obs.get(mask_key) if mask_key in obs else None
                normalized_obs[key] = self._obs_normalizer[key].normalize(val, mask)
            else:
                normalized_obs[key] = val  # Pass MultiBinary or other non-Box types through

        info['original_obs'] = obs
        return normalized_obs, info

    def save(self) -> Dict[str, Any]:
        """Save the observation normalizers."""
        saved = self.env.save() if hasattr(self.env, 'save') else {}
        saved['obs_normalizer'] = self._obs_normalizer
        return saved
    

class DictUnsqueeze(gym.Wrapper):
    """Unsqueeze the observation, reward, cost, terminated, truncated, and info for dictionary-type observations.

    Examples:
        >>> env = Unsqueeze(env, device=torch.device('cpu'))
    """

    def __init__(self, env: CMDP, device: torch.device) -> None:
        """Initialize an instance of :class:`Unsqueeze`."""
        super().__init__(env=env)
        self._device = device
        assert self.num_envs == 1, 'Unsqueeze only works with single environment'
        assert isinstance(self.observation_space, spaces.Dict), 'Observation space must be Dict'

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The vector information will be unsqueezed to [1, dim1, dim2, ...] for agent training.

        Args:
            action (torch.Tensor): The action from the agent or random, expected shape [1, action_dim].

        Returns:
            observation: The agent's observation dictionary with unsqueezed tensors.
            reward: The amount of reward returned after previous action, shape [1].
            cost: The amount of cost returned after previous action, shape [1].
            terminated: Whether the episode has ended, shape [1].
            truncated: Whether the episode has been truncated, shape [1].
            info: Information logged by the environment with unsqueezed tensors.
        """
        # Squeeze action from [1, action_dim] to [action_dim] for the environment
        action = action.squeeze(0).to(self._device)
        obs, reward, cost, terminated, truncated, info = super().step(action)

        # Unsqueeze observation dictionary
        obs = {key: val.unsqueeze(0).to(self._device) for key, val in obs.items()}

        # Unsqueeze scalar outputs
        reward = reward.unsqueeze(0).to(self._device)
        cost = cost.unsqueeze(0).to(self._device)
        terminated = terminated.unsqueeze(0).to(self._device)
        truncated = truncated.unsqueeze(0).to(self._device)

        # Unsqueeze tensors in info dictionary
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0).to(self._device)

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Reset the environment and return a new observation.

        .. note::
            The vector information will be unsqueezed to [1, dim1, dim2, ...] for agent training.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): Options for the environment. Defaults to None.

        Returns:
            observation: The initial observation dictionary with unsqueezed tensors.
            info: Information logged by the environment with unsqueezed tensors.
        """
        obs, info = super().reset(seed=seed, options=options)

        # Unsqueeze observation dictionary
        obs = {key: val.unsqueeze(0).to(self._device) for key, val in obs.items()}

        # Unsqueeze tensors in info dictionary
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0).to(self._device)

        return obs, info