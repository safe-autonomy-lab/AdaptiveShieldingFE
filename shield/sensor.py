from dataclasses import dataclass

import torch



@dataclass
class LidarConfig:
    r"""Lidar observation parameters.

    Attributes:
        num_bins (int): Bins (around a full circle) for lidar sensing.
        max_dist (float): Maximum distance for lidar sensitivity (if None, exponential distance).
        exp_gain (float): Scaling factor for distance in exponential distance lidar.
        type (str): 'pseudo', 'natural', see self._obs_lidar().
        alias (bool): Lidar bins alias into each other.
    """

    num_bins: int = 16
    max_dist: float = 3
    exp_gain: float = 1.0
    type: str = 'pseudo'
    alias: bool = True


class LidarSensorProcessor:
    """Process lidar sensor data for agent navigation with obstacle detection."""
    
    def __init__(self, lidar_conf = LidarConfig(), radius: float = 1.0):
        """Initialize lidar sensor processor.
        
        Args:
            lidar_conf: Lidar configuration parameters
            radius: Agent sensing radius
        """
        self.lidar_conf = lidar_conf
        self.radius = radius
        
        assert self.lidar_conf.num_bins > 0, "Number of bins must be positive"
        assert self.radius >= 0, "Radius must be non-negative"

    def _ego_xy(self, pos: torch.Tensor, agent_pos: torch.Tensor,
                agent_mat: torch.Tensor) -> torch.Tensor:
        """Return egocentric XY vector to a position from the agent.
        Args:
            pos: Target position [x, y]
            agent_pos: Agent position [x, y, z]
            agent_mat: Agent rotation matrix [3, 3]
        Returns:
            Egocentric XY coordinates
        """
        agent_pos = agent_pos.squeeze()
        agent_mat = agent_mat.squeeze()
        pos_3vec = torch.cat([pos, torch.tensor([0.0], dtype=pos.dtype, device=pos.device)])
        world_3vec = pos_3vec - agent_pos
        return torch.matmul(world_3vec, agent_mat)[:2]

    def _obs_lidar_pseudo(self, positions: torch.Tensor, agent_pos: torch.Tensor,
                        agent_mat: torch.Tensor) -> torch.Tensor:
        """Process batched lidar observations.
        Args:
            positions: Shape (num_env, num_points, 3) or (num_env, num_points, 2)
            agent_pos: Shape (num_env, 3)
            agent_mat: Shape (num_env, 3, 3)
        Returns:
            Processed lidar observations of shape (num_env, num_bins)
        """
        num_env = agent_pos.shape[0]
        all_obs = []
        for env_idx in range(num_env):
            env_positions = positions[env_idx]
            env_agent_pos = agent_pos[env_idx]
            env_agent_mat = agent_mat[env_idx]
            if env_positions.ndim == 1:
                env_positions = env_positions.unsqueeze(0)
            obs = torch.zeros(self.lidar_conf.num_bins, dtype=positions.dtype, device=positions.device)
            for pos in env_positions:
                if pos.shape == (3,):
                    pos = pos[:2]
                assert pos.shape == (2,), f'Bad pos {pos}'
                ego = self._ego_xy(pos, env_agent_pos, env_agent_mat)
                z = torch.complex(ego[0], ego[1])
                dist = torch.abs(z)
                angle = torch.angle(z) % (torch.pi * 2)
                bin_size = (torch.pi * 2) / self.lidar_conf.num_bins
                bin_idx = int((angle / bin_size).item())
                if bin_idx > self.lidar_conf.num_bins - 1:
                    continue
                bin_angle = bin_size * bin_idx
                if self.lidar_conf.type == 'pseudo':
                    if self.lidar_conf.max_dist is not None:
                        sensor = max(0.0, self.lidar_conf.max_dist - dist.item()) / self.lidar_conf.max_dist
                        sensor = torch.tensor(sensor, dtype=obs.dtype, device=obs.device)
                    else:
                        sensor = torch.exp(-self.lidar_conf.exp_gain * dist)
                else:
                    # Natural lidar - exponential decay
                    sensor = torch.exp(-self.lidar_conf.exp_gain * dist)
                obs[bin_idx] = torch.max(obs[bin_idx], sensor)
                if self.lidar_conf.alias:
                    alias = ((angle - bin_angle) / bin_size).item()
                    assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin_idx}'
                    bin_plus = (bin_idx + 1) % self.lidar_conf.num_bins
                    bin_minus = (bin_idx - 1) % self.lidar_conf.num_bins
                    obs[bin_plus] = torch.max(obs[bin_plus], torch.tensor(alias, dtype=obs.dtype, device=obs.device) * sensor)
                    obs[bin_minus] = torch.max(obs[bin_minus], torch.tensor(1 - alias, dtype=obs.dtype, device=obs.device) * sensor)
            all_obs.append(obs)
        return torch.stack(all_obs)

    def _get_agent_pos_and_matrix(self, robot_predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get agent position and rotation matrix from robot predictions.
        Args:
            robot_predictions: Robot state predictions [num_env, 6 + robot_dim]
        Returns:
            agent_pos: Agent position [num_env, 3]
            agent_mat: Agent rotation matrix [num_env, 3, 3]
        """
        # Extract and pad agent position with z coordinate
        num_env = robot_predictions.shape[0]
        agent_pos = torch.cat([robot_predictions[:, :2].clone(), torch.full((num_env, 1), 0.1, dtype=robot_predictions.dtype, device=robot_predictions.device)], dim=1)
        # Convert 2x2 rotation matrix to 3x3
        agent_mat = robot_predictions[:, 2:6].reshape(-1, 2, 2)
        agent_mat_3x3 = torch.zeros((agent_mat.shape[0], 3, 3), dtype=agent_mat.dtype, device=agent_mat.device)
        agent_mat_3x3[:, :2, :2] = agent_mat
        agent_mat_3x3[:, 2, 2] = 1.0
        return agent_pos, agent_mat_3x3

    def _process_lidar_observations(self,
                                    obs: torch.Tensor,
                                    agent_pos: torch.Tensor,
                                    agent_mat: torch.Tensor,
                                    z_values: float = 0.0
                                    ) -> torch.Tensor:
        """Process lidar observations for different environment elements.
        Args:
            obs: Raw observations of shape (num_env, num_points, 2) or (num_env, 2)
            agent_pos: Agent position [num_env, 3]
            agent_mat: Agent rotation matrix [num_env, 3, 3]
            z_values: Z-coordinate value to pad with
        Returns:
            Processed lidar observations
        """
        import torch.nn.functional as F
        if obs.ndim == 2:  # Shape: (num_env, 2)
            # For single observations, reshape and pad
            obs = obs.reshape(-1, 2)  # Ensure (num_env, 2) shape
            obs = F.pad(obs, (0, 1), mode='constant', value=z_values)
            return self._obs_lidar_pseudo(obs, agent_pos, agent_mat)
        else:  # Shape: (num_env, num_points, 2)
            obs = F.pad(obs, (0, 0, 0, 1), mode='constant', value=z_values)
            return self._obs_lidar_pseudo(obs, agent_pos, agent_mat)