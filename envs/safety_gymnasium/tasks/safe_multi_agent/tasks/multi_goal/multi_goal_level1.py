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
"""Multi Goal level 1."""

from envs.safety_gymnasium.tasks.safe_multi_agent.assets.free_geoms import Vases
from envs.safety_gymnasium.tasks.safe_multi_agent.assets.geoms import Hazards
from envs.safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal.multi_goal_level0 import (
    MultiGoalLevel0,
)


class MultiGoalLevel1(MultiGoalLevel0):
    """An agent must navigate to a goal while avoiding hazards.

    One vase is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(Hazards(num=8, keepout=0.18))
        self._add_free_geoms(Vases(num=1, is_constrained=False))
        self.contact_other_cost = 1.0
