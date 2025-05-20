# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Gremlin."""

from dataclasses import dataclass, field

import numpy as np

from envs.safety_gymnasium.assets.color import COLOR
from envs.safety_gymnasium.assets.group import GROUP
from envs.safety_gymnasium.bases.base_object import Mocap

MOVING_OBJECTS = 'social-force'

@dataclass
class Gremlins(Mocap):  # pylint: disable=too-many-instance-attributes
    """Gremlins (moving objects we should avoid)"""

    name: str = 'gremlins'
    num: int = 0  # Number of gremlins in the world
    size: float = 0.1
    placements: list = None  # Gremlins placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.5  # Radius for keeping out (contains gremlin path)
    travel: float = 0.3  # Radius of the circle traveled in
    contact_cost: float = 1.0  # Cost for touching a gremlin
    dist_threshold: float = 0.2  # Threshold for cost for being too close
    dist_cost: float = 1.0  # Cost for being within distance threshold
    density: float = 0.001

    color: np.array = COLOR['gremlin']
    alpha: float = 1
    group: np.array = GROUP['gremlin']
    is_lidar_observed: bool = True
    is_constrained: bool = True
    is_meshed: bool = False
    mesh_name: str = name[:-1]

    mobstalces: list = field(default_factory=list)
    velocities: dict = field(default_factory=dict)
    x_limit: float = 2.0
    y_limit: float = 2.0
    goal_radius: float = 0.1  # Distance threshold to consider goal reached

    def __post_init__(self):
        """Initialize basic attributes but defer mobstalces initialization."""
        self.mobstalces = []  # Just initialize empty list
        self._initialized = False  # Flag to track initialization state

    def _initialize_mobstalces(self):
        """Initialize pedestrians once engine is ready."""
        for i in range(self.num):
            name = f'gremlin{i}obj'
            initial_pos = np.array([0, 0])
            
            # max_speed = np.random.uniform(0.1, 1.0)
            max_speed = np.random.uniform(0.1, 0.5)
            self.mobstalces.append({
                'name': f'gremlin{i}',
                'x': initial_pos[0],
                'y': initial_pos[1],
                'velocity': np.zeros(2),
                'max_speed': max_speed,
                'goal_x': None,
                'goal_y': None,
                'orientation': 0.0
            })
            
            goal_x, goal_y = self._generate_new_goal(initial_pos)
            self.mobstalces[i]['goal_x'] = goal_x
            self.mobstalces[i]['goal_y'] = goal_y
            # Format: [x, y, vel_x, vel_y, orientation]
            self.velocities[name] = np.zeros(3)

    def _generate_new_goal(self, current_pos):
        """Generate a new goal position within boundaries.
        
        Args:
            current_pos: numpy array [x, y] of current position
        Returns:
            tuple: (goal_x, goal_y)
        """
        # Generate goal within boundaries with some margin
        margin = 0.1  # Prevent goals too close to boundaries
        max_attempts = 10
        
        for _ in range(max_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(1.0, 2.0)
            new_goal = current_pos + np.array([
                np.cos(angle) * distance,
                np.sin(angle) * distance
            ])
            
            # Check if the goal is within boundaries
            if (abs(new_goal[0]) <= (self.x_limit - margin) and 
                abs(new_goal[1]) <= (self.y_limit - margin)):
                return new_goal[0], new_goal[1]
        
        # If we couldn't find a valid goal, generate one towards the center
        direction_to_center = -current_pos
        if np.linalg.norm(direction_to_center) > 0:
            direction_to_center = direction_to_center / np.linalg.norm(direction_to_center)
            new_goal = current_pos + direction_to_center * distance
            return new_goal[0], new_goal[1]
        
        # Fallback to center with small random offset
        return (np.random.uniform(-0.1, 0.1), 
                np.random.uniform(-0.1, 0.1))
    

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object"""
        return {'obj': self.get_obj(xy_pos, rot), 'mocap': self.get_mocap(xy_pos, rot)}

    def get_obj(self, xy_pos, rot):
        """To facilitate get objects config for this object"""
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, self.size],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': np.ones(3) * self.size,
                    'type': 'box',
                    'density': self.density,
                    'group': self.group,
                    'rgba': self.color * np.array([1, 1, 1, self.alpha]),
                },
            ],
        }
        if self.is_meshed:
            body['geoms'][0].update(
                {
                    'type': 'mesh',
                    'mesh': self.mesh_name,
                    'material': self.mesh_name,
                    'rgba': np.array([1, 1, 1, 1]),
                    'euler': [np.pi / 2, 0, 0],
                },
            )
            body['pos'][2] = 0.0
        return body

    def get_mocap(self, xy_pos, rot):
        """To facilitate get mocaps config for this object"""
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, self.size],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': np.ones(3) * self.size,
                    'type': 'box',
                    'group': self.group,
                    # 'rgba': self.color * np.array([1, 1, 1, self.alpha]),
                    'rgba': self.color * np.array([1, 1, 1, 0]),
                },
            ],
        }
        if self.is_meshed:
            body['geoms'][0].update(
                {
                    'type': 'mesh',
                    'mesh': self.mesh_name,
                    'material': self.mesh_name,
                    'rgba': np.array([1, 1, 1, 0]),
                    'euler': [np.pi / 2, 0, 0],
                },
            )
            body['pos'][2] = 0.0
        return body

    def cal_cost(self):
        """Contacts processing."""
        cost = {}
        if not self.is_constrained:
            return cost
        cost['cost_gremlins'] = 0
        for contact in self.engine.data.contact[: self.engine.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.engine.model.geom(g).name for g in geom_ids])
            if any(n.startswith('gremlin') for n in geom_names) and any(
                n in self.agent.body_info.geom_names for n in geom_names
            ):
                # pylint: disable-next=no-member
                cost['cost_gremlins'] += self.contact_cost

        return cost
    
    def _calculate_periodic_distance_and_direction(self, pos1, pos2):
        """Calculate distance and direction considering periodic boundaries.
        
        Args:
            pos1: Current position numpy array [x, y]
            pos2: Target position numpy array [x, y]
            
        Returns:
            tuple: (distance, direction_vector)
        """
        # Calculate direct difference
        diff = pos2 - pos1
        
        # Check for shorter path across periodic boundaries
        for i in range(2):  # For both x and y dimensions
            limit = self.x_limit if i == 0 else self.y_limit
            while diff[i] > limit:
                diff[i] -= 2 * limit
            while diff[i] < -limit:
                diff[i] += 2 * limit
        
        distance = np.linalg.norm(diff)
        direction = diff / distance if distance > 0 else np.zeros(2)
        
        return distance, direction
    
    def move(self):
        """Set mocap object positions before a physics step is executed."""
        if not self._initialized:
            self._initialize_mobstalces()
            self._initialized = True
            return
        
        dt = 0.002  # Simulation timestep
        mass = 80  # kg, average human mass
        tau = 0.5  # Relaxation time
        
        # Social force parameters
        A = 2000  # N (repulsion strength)
        B = 0.08  # m (repulsion range)
        
        # Boundary force parameters
        boundary_force_strength = 5.0
        boundary_range = 0.2
        
        # Update positions and observations for all pedestrians
        self.velocities = {}
        for i in range(self.num):
            mobstacle = self.mobstalces[i]
            name = f'gremlin{i}obj'
            current_pos = np.array([mobstacle['x'], mobstacle['y']])
            goal_pos = np.array([mobstacle['goal_x'], mobstacle['goal_y']])
            
            # Calculate distance and direction to goal
            diff_to_goal = goal_pos - current_pos
            distance_to_goal = np.linalg.norm(diff_to_goal)

            if MOVING_OBJECTS == 'random':
                if distance_to_goal < self.goal_radius:
                    mobstacle['goal_x'], mobstacle['goal_y'] = self._generate_new_goal(current_pos)
                    continue
                
                # Calculate desired orientation towards goal
                desired_orientation = np.arctan2(diff_to_goal[1], diff_to_goal[0])
                
                # Add noise to orientation
                angle_noise_std = 0.1
                orientation_noise = np.random.normal(0, angle_noise_std)
                mobstacle['orientation'] = desired_orientation + orientation_noise
                
                # Calculate velocity with noise
                pos_noise_std = 0.01
                position_noise = np.random.normal(0, pos_noise_std)
                velocity = mobstacle['max_speed'] + position_noise
                cos_orientation = np.cos(mobstacle['orientation'])
                sin_orientation = np.sin(mobstacle['orientation'])
                # Update velocity and position
                mobstacle['velocity'] = np.array([
                    velocity * cos_orientation,
                    velocity * sin_orientation
                ])
                
                new_pos = current_pos + mobstacle['velocity'] * dt
                
                # Ensure position stays within boundaries
                new_pos[0] = np.clip(new_pos[0], -self.x_limit, self.x_limit)
                new_pos[1] = np.clip(new_pos[1], -self.y_limit, self.y_limit)
                
                # Update pedestrian state
                mobstacle['x'] = new_pos[0]
                mobstacle['y'] = new_pos[1]
                
                # Set mocap position
                pos = np.r_[new_pos, [self.size]]
                self.set_mocap_pos(mobstacle['name'] + 'mocap', pos)

                # Update observations in the same format as humans
                # [x, y, vel, cos_orientation, sin_orientation]
                self.velocities[name] = np.array([
                    velocity,
                    cos_orientation,
                    sin_orientation
                ])

            elif MOVING_OBJECTS == 'social-force':
                direction_to_goal = diff_to_goal / distance_to_goal if distance_to_goal > 0 else np.zeros(2)
                # Check if goal is reached
                if distance_to_goal < self.goal_radius:
                    # Generate new goal within boundaries
                    mobstacle['goal_x'], mobstacle['goal_y'] = self._generate_new_goal(current_pos)
                    goal_pos = np.array([mobstacle['goal_x'], mobstacle['goal_y']])
                    diff_to_goal = goal_pos - current_pos
                    distance_to_goal = np.linalg.norm(diff_to_goal)
                    direction_to_goal = diff_to_goal / distance_to_goal if distance_to_goal > 0 else np.zeros(2)
                  # 1. Calculate desired force (towards goal)
                desired_velocity = direction_to_goal * mobstacle['max_speed']
                desired_force = mass * (desired_velocity - mobstacle['velocity']) / tau
                
                # 2. Calculate social force (repulsion from other pedestrians)
                social_force = np.zeros(2)
                for j in range(self.num):
                    if i != j:
                        other_pos = np.array([self.mobstalces[j]['x'], self.mobstalces[j]['y']])
                        diff = current_pos - other_pos
                        distance = np.linalg.norm(diff)
                        if distance < 2:  # Only consider nearby pedestrians
                            direction = diff / distance if distance > 0 else np.zeros(2)
                            social_force += A * np.exp(-distance/B) * direction
                
                # 3. Calculate boundary force (repulsion from boundaries)
                boundary_force = np.zeros(2)
                
                # X-boundary forces
                if abs(current_pos[0]) > (self.x_limit - boundary_range):
                    force_magnitude = boundary_force_strength * np.exp(
                        -(self.x_limit - abs(current_pos[0])) / boundary_range
                    )
                    boundary_force[0] = -np.sign(current_pos[0]) * force_magnitude
                
                # Y-boundary forces
                if abs(current_pos[1]) > (self.y_limit - boundary_range):
                    force_magnitude = boundary_force_strength * np.exp(
                        -(self.y_limit - abs(current_pos[1])) / boundary_range
                    )
                    boundary_force[1] = -np.sign(current_pos[1]) * force_magnitude
                
                # 4. Sum all forces and update velocity
                total_force = desired_force + social_force + boundary_force
                mobstacle['velocity'] += (total_force / mass) * dt
                
                # Limit maximum speed
                speed = np.linalg.norm(mobstacle['velocity'])
                if speed > mobstacle['max_speed']:
                    mobstacle['velocity'] *= mobstacle['max_speed'] / speed
                
                # Update position
                new_pos = current_pos + mobstacle['velocity'] * dt
                
                # Ensure position stays within boundaries
                new_pos[0] = np.clip(new_pos[0], -self.x_limit, self.x_limit)
                new_pos[1] = np.clip(new_pos[1], -self.y_limit, self.y_limit)
                
                # Update pedestrian state
                mobstacle['x'] = new_pos[0]
                mobstacle['y'] = new_pos[1]
                
                # Set mocap position
                pos = np.r_[new_pos, [self.size]]
                self.set_mocap_pos(mobstacle['name'] + 'mocap', pos)
                self.velocities[mobstacle['name'] + 'obj'] = mobstacle['velocity']

    @property
    def pos(self):
        """Helper to get the current gremlin position."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'gremlin{i}obj').xpos.copy() for i in range(self.num)]
