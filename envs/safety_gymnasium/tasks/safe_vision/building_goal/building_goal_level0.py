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
"""Building Goal level 0."""

from envs.safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0


class BuildingGoalLevel0(GoalLevel0):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.floor_conf.size = [100, 100, 0.1]

        for obj in self._obstacles:
            obj.is_meshed = True
        self._is_load_static_geoms = True
