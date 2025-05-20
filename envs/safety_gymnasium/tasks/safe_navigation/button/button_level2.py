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
"""Button task 1."""

from envs.safety_gymnasium.assets.geoms import Hazards
from envs.safety_gymnasium.tasks.safe_navigation.button.button_level0 import ButtonLevel0


class ButtonLevel2(ButtonLevel0):
    """An agent must press a goal button while avoiding hazards and gremlins.

    And while not pressing any of the wrong buttons.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self._add_geoms(Hazards(num=8, keepout=0.18))
        self.buttons.is_constrained = False # pylint: disable=no-member