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
"""Custom goal level 1."""  


from envs.safety_gymnasium.assets.geoms import Hazards, Pillars, Apples, Oranges
from envs.safety_gymnasium.assets.free_geoms import Vases
from envs.safety_gymnasium.assets.mocaps import Gremlins
from envs.safety_gymnasium.tasks.safe_navigation.custom_goal.custom_goal_level0 import CustomGoalLevel0


class CustomGoalLevel1(CustomGoalLevel0):
    """An agent must navigate to a goal while obstacles customized.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        nbr_of_hazards = config.get('NBR_OF_HAZARDS', 0)
        nbr_of_pillars = config.get('NBR_OF_PILLARS', 0)
        nbr_of_apples = config.get('NBR_OF_APPLES', 0)
        nbr_of_oranges = config.get('NBR_OF_ORANGES', 0)
        nbr_of_gremlins = config.get('NBR_OF_GREMLINS', 0)
        nbr_of_vases = config.get('NBR_OF_VASES', 0)
        placement_extents = config.get('PLACEMENT_EXTENTS', 2)

        self.placements_conf.extents = [-placement_extents, -placement_extents, placement_extents, placement_extents]
        if nbr_of_hazards > 0:
            self._add_geoms(Hazards(num=nbr_of_hazards, keepout=0.25))
        if nbr_of_pillars > 0:
            self._add_geoms(Pillars(num=nbr_of_pillars, keepout=0.25))
        if nbr_of_apples > 0:
            self._add_geoms(Apples(num=nbr_of_apples, keepout=0.25))
        if nbr_of_oranges > 0:
            self._add_geoms(Oranges(num=nbr_of_oranges, keepout=0.25))
        if nbr_of_vases:
            self._add_free_geoms(Vases(num=nbr_of_vases, keepout=0.25))
        if nbr_of_gremlins > 0:
            self._add_mocaps(Gremlins(num=nbr_of_gremlins, keepout=0.25))

