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
"""World."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

import mujoco
import numpy as np
import xmltodict
import yaml

# Set up logging for parameter manipulation
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

import envs.safety_gymnasium
from envs.safety_gymnasium.utils.common_utils import build_xml_from_dict, convert, rot2quat
from envs.safety_gymnasium.utils.task_utils import get_body_xvelp

from .base_world import BaseWorld


# Default location to look for xmls folder:
BASE_DIR = os.path.dirname(envs.safety_gymnasium.__file__)


@dataclass
class Engine:
    """Physical engine."""

    # pylint: disable=no-member
    model: mujoco.MjModel = None
    data: mujoco.MjData = None

    def update(self, model, data):
        """Set engine."""
        self.model = model
        self.data = data


class World(BaseWorld):  # pylint: disable=too-many-instance-attributes
    """This class starts mujoco simulation.

    And contains some apis for interacting with mujoco."""

    # Default configuration (this should not be nested since it gets copied)
    # *NOTE:* Changes to this configuration should also be reflected in `Builder` configuration
    DEFAULT: ClassVar[dict[str, Any]] = {
        'agent_base': 'assets/xmls/car.xml',  # Which agent XML to use as the base
        'agent_xy': np.zeros(2),  # agent XY location
        'agent_rot': 0,  # agent rotation about Z axis
        'floor_size': [3.5, 3.5, 0.1],  # Used for displaying the floor
        # FreeGeoms -- this is processed and added by the Builder class
        'free_geoms': {},  # map from name -> object dict
        # Geoms -- similar to objects, but they are immovable and fixed in the scene.
        'geoms': {},  # map from name -> geom dict
        # Mocaps -- mocap objects which are used to control other objects
        'mocaps': {},
        'floor_type': 'mat',
        'task_name': None,
        'env_config': None,
        # Hidden parameter configuration
        'min_mult': 0.7,
        'max_mult': 1.3,
        'fix_hidden_parameters': False,
        'is_out_of_distribution': False,
        'hidden_param_dims': 2,
    }

    def __init__(self, agent, obstacles, config=None) -> None:
        """config - JSON string or dict of configuration.  See self.parse()"""
        
        # Initialize attributes to None first
        self.agent_base_path = None
        self.agent_base_xml = None
        self.xml = None
        self.xml_string = None
        # This is for changing the underyling parameters of the environment
        self.fix_hidden_parameters = None
        self.hidden_param_dims = None
        self.is_out_of_distribution = None
        self.min_mult = None
        self.max_mult = None
        self.rule_out_param = ""

        if config:
            self.parse(config)  # Parse configuration - this will set proper values

        self.first_reset = True

        self._agent = agent  # pylint: disable=no-member
        self._obstacles = obstacles
        # Since two times of calling this function, we need to reset the episode count

        self.engine = Engine()
        self.bind_engine()

    def parse(self, config):
        """Parse a config dict - see self.DEFAULT for description."""
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key: {str(key)}. Available keys: {list(self.DEFAULT.keys())}'
            setattr(self, key, value)

    def bind_engine(self):
        """Send the new engine instance to the agent and obstacles."""
        self._agent.set_engine(self.engine)
        for obstacle in self._obstacles:
            obstacle.set_engine(self.engine)

    def setup_hidden_parameters(self, fix_hidden_parameters: bool = False, is_out_of_distribution: bool = False, hidden_param_dims: int = 2):
        """Set the hidden parameters for the world."""
        self.fix_hidden_parameters = fix_hidden_parameters
        self.hidden_param_dims = hidden_param_dims
    
    def set_parameters_range(self, min_mult: float, max_mult: float):
        """Set the parameters range for the world."""
        self.min_mult = min_mult
        self.max_mult = max_mult

    def build(self):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        # episode count will be set outside of the environment
        """Build a world, including generating XML and moving objects."""
        # Read in the base XML (contains agent, camera, floor, etc)
        self.agent_base_path = os.path.join(BASE_DIR, self.agent_base)  # pylint: disable=no-member
        with open(self.agent_base_path, encoding='utf-8') as f:  # pylint: disable=invalid-name
            self.agent_base_xml = f.read()

        self.xml = xmltodict.parse(self.agent_base_xml)  # Nested OrderedDict objects
                    
        if self.task_name in ['FormulaOne']:  # pylint: disable=no-member
            self.xml['mujoco']['option']['@integrator'] = 'RK4'
            self.xml['mujoco']['option']['@timestep'] = '0.004'

        if 'compiler' not in self.xml['mujoco']:
            compiler = xmltodict.parse(
                f"""<compiler
                angle="radian"
                meshdir="{BASE_DIR}/assets/meshes"
                texturedir="{BASE_DIR}/assets/textures"
                />""",
            )
            self.xml['mujoco']['compiler'] = compiler['compiler']
        else:
            self.xml['mujoco']['compiler'].update(
                {
                    '@angle': 'radian',
                    '@meshdir': os.path.join(BASE_DIR, 'assets', 'meshes'),
                    '@texturedir': os.path.join(BASE_DIR, 'assets', 'textures'),
                },
            )

        # Convenience accessor for xml dictionary
        worldbody = self.xml['mujoco']['worldbody']

        # Move agent position to starting position
        worldbody['body']['@pos'] = convert(
            # pylint: disable-next=no-member
            np.r_[self.agent_xy, self._agent.z_height],
        )
        worldbody['body']['@quat'] = convert(rot2quat(self.agent_rot))  # pylint: disable=no-member

        # We need this because xmltodict skips over single-item lists in the tree
        worldbody['body'] = [worldbody['body']]
        if 'geom' in worldbody:
            worldbody['geom'] = [worldbody['geom']]
        else:
            worldbody['geom'] = []
        # Add equality section if missing
        if 'equality' not in self.xml['mujoco']:
            self.xml['mujoco']['equality'] = OrderedDict()
        equality = self.xml['mujoco']['equality']
        if 'weld' not in equality:
            equality['weld'] = []

        # Add asset section if missing
        if 'asset' not in self.xml['mujoco']:
            self.xml['mujoco']['asset'] = {}
        if 'texture' not in self.xml['mujoco']['asset']:
            self.xml['mujoco']['asset']['texture'] = []
        if 'material' not in self.xml['mujoco']['asset']:
            self.xml['mujoco']['asset']['material'] = []
        if 'mesh' not in self.xml['mujoco']['asset']:
            self.xml['mujoco']['asset']['mesh'] = []
        material = self.xml['mujoco']['asset']['material']
        texture = self.xml['mujoco']['asset']['texture']
        mesh = self.xml['mujoco']['asset']['mesh']

        # load all assets config from .yaml file
        with open(os.path.join(BASE_DIR, 'configs/assets.yaml'), encoding='utf-8') as file:
            assets_config = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506

        texture.append(assets_config['textures']['skybox'])

        if self.floor_type == 'mat':  # pylint: disable=no-member
            texture.append(assets_config['textures']['matplane'])
            material.append(assets_config['materials']['matplane'])
        elif self.floor_type == 'village':  # pylint: disable=no-member
            texture.append(assets_config['textures']['village_floor'])
            material.append(assets_config['materials']['village_floor'])
        elif self.floor_type == 'mud':  # pylint: disable=no-member
            texture.append(assets_config['textures']['mud_floor'])
            material.append(assets_config['materials']['mud_floor'])
        elif self.floor_type == 'none':  # pylint: disable=no-member
            self.floor_size = [1e-9, 1e-9, 0.1]  # pylint: disable=attribute-defined-outside-init
        else:
            raise NotImplementedError

        selected_textures = {}
        selected_materials = {}
        selected_meshes = {}
        for config in (
            # pylint: disable=no-member
            list(self.geoms.values())
            + list(self.free_geoms.values())
            + list(self.mocaps.values())
            # pylint: enable=no-member
        ):
            if 'type' not in config:
                for geom in config['geoms']:
                    if geom['type'] != 'mesh':
                        continue
                    mesh_name = geom['mesh']
                    if mesh_name in assets_config['textures']:
                        selected_textures[mesh_name] = assets_config['textures'][mesh_name]
                        selected_materials[mesh_name] = assets_config['materials'][mesh_name]
                    selected_meshes[mesh_name] = assets_config['meshes'][mesh_name]
            elif config['type'] == 'mesh':
                mesh_name = config['mesh']
                if mesh_name in assets_config['textures']:
                    selected_textures[mesh_name] = assets_config['textures'][mesh_name]
                    selected_materials[mesh_name] = assets_config['materials'][mesh_name]
                selected_meshes[mesh_name] = assets_config['meshes'][mesh_name]
        texture += selected_textures.values()
        material += selected_materials.values()
        mesh += selected_meshes.values()

        # Add light to the XML dictionary
        light = xmltodict.parse(
            """<b>
            <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true"
                exponent="1" pos="0 0 0.5" specular="0 0 0" castshadow="false"/>
            </b>""",
        )
        worldbody['light'] = light['b']['light']

        # Add floor to the XML dictionary if missing
        if not any(g.get('@name') == 'floor' for g in worldbody['geom']):
            floor = xmltodict.parse(
                """
                <geom name="floor" type="plane" condim="6"/>
                """,
            )
            worldbody['geom'].append(floor['geom'])

        # Make sure floor renders the same for every world
        for g in worldbody['geom']:  # pylint: disable=invalid-name
            if g['@name'] == 'floor':
                g.update(
                    {
                        '@size': convert(self.floor_size),  # pylint: disable=no-member
                        '@rgba': '1 1 1 1',
                    },
                )
                if self.floor_type == 'mat':  # pylint: disable=no-member
                    g.update({'@material': 'matplane'})
                elif self.floor_type == 'village':  # pylint: disable=no-member
                    g.update({'@material': 'village_floor'})
                elif self.floor_type == 'mud':  # pylint: disable=no-member
                    g.update({'@material': 'mud_floor'})
                elif self.floor_type == 'none':  # pylint: disable=no-member
                    pass
                else:
                    raise NotImplementedError
        # Add cameras to the XML dictionary
        cameras = xmltodict.parse(
            """<b>
            <camera name="fixednear" pos="0 -2 2" zaxis="0 -1 1"/>
            <camera name="fixedfar" pos="0 -5 5" zaxis="0 -1 1"/>
            <camera name="fixedfar++" pos="0 -10 10" zaxis="0 -1 1"/>
            </b>""",
        )
        worldbody['camera'] = cameras['b']['camera']

        # Build and add a tracking camera (logic needed to ensure orientation correct)
        theta = self.agent_rot + np.pi  # pylint: disable=no-member
        xyaxes = {
            'x1': np.cos(theta),
            'x2': -np.sin(theta),
            'x3': 0,
            'y1': np.sin(theta),
            'y2': np.cos(theta),
            'y3': 1,
        }
        pos = {
            'xp': 0 * np.cos(theta) + (-2) * np.sin(theta),
            'yp': 0 * (-np.sin(theta)) + (-2) * np.cos(theta),
            'zp': 2,
        }
        track_camera = xmltodict.parse(
            """<b>
            <camera name="track" mode="track" pos="{xp} {yp} {zp}"
                xyaxes="{x1} {x2} {x3} {y1} {y2} {y3}"/>
            </b>""".format(
                **pos,
                **xyaxes,
            ),
        )
        if 'camera' in worldbody['body'][0]:
            if isinstance(worldbody['body'][0]['camera'], list):
                worldbody['body'][0]['camera'] = worldbody['body'][0]['camera'] + [
                    track_camera['b']['camera'],
                ]
            else:
                worldbody['body'][0]['camera'] = [
                    worldbody['body'][0]['camera'],
                    track_camera['b']['camera'],
                ]
        else:
            worldbody['body'][0]['camera'] = [
                track_camera['b']['camera'],
            ]

        # Add free_geoms to the XML dictionary
        for name, object in self.free_geoms.items():  # pylint: disable=redefined-builtin, no-member
            assert object['name'] == name, f'Inconsistent {name} {object}'
            object = object.copy()  # don't modify original object
            object['freejoint'] = object['name']
            if name == 'push_box':
                object['quat'] = rot2quat(object.pop('rot'))
                dim = object['geoms'][0]['size'][0]
                object['geoms'][0]['dim'] = dim
                object['geoms'][0]['width'] = dim / 2
                object['geoms'][0]['x'] = dim
                object['geoms'][0]['y'] = dim
                # pylint: disable-next=consider-using-f-string
                collision_xml = """
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" density="{density}"
                            rgba="{rgba}" group="{group}"/>
                        <geom name="col1" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} {y} 0"/>
                        <geom name="col2" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} {y} 0"/>
                        <geom name="col3" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} -{y} 0"/>
                        <geom name="col4" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} -{y} 0"/>
                        """.format(
                    **{k: convert(v) for k, v in object['geoms'][0].items()},
                )
                if len(object['geoms']) == 2:
                    # pylint: disable-next=consider-using-f-string
                    visual_xml = """
                        <geom name="{name}" type="mesh" mesh="{mesh}" material="{material}" pos="{pos}"
                        rgba="1 1 1 1" group="{group}" contype="{contype}" conaffinity="{conaffinity}" density="{density}"
                        euler="{euler}"/>
                    """.format(
                        **{k: convert(v) for k, v in object['geoms'][1].items()},
                    )
                else:
                    visual_xml = """"""
                body = xmltodict.parse(
                    # pylint: disable-next=consider-using-f-string
                    f"""
                    <body name="{object['name']}" pos="{convert(object['pos'])}" quat="{convert(object['quat'])}">
                        {collision_xml}
                        {visual_xml}
                    </body>
                """,
                )
            else:
                if object['geoms'][0]['type'] == 'mesh':
                    object['geoms'][0]['condim'] = 6
                object['quat'] = rot2quat(object.pop('rot'))
                body = build_xml_from_dict(object)
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])
        # Add mocaps to the XML dictionary
        for name, mocap in self.mocaps.items():  # pylint: disable=no-member
            # Mocap names are suffixed with 'mocap'
            assert mocap['name'] == name, f'Inconsistent {name}'
            assert (
                name.replace('mocap', 'obj') in self.free_geoms  # pylint: disable=no-member
            ), f'missing object for {name}'  # pylint: disable=no-member
            # Add the object to the world
            mocap = mocap.copy()  # don't modify original object
            mocap['quat'] = rot2quat(mocap.pop('rot'))
            mocap['mocap'] = 'true'
            mocap['geoms'][0]['contype'] = 0
            mocap['geoms'][0]['conaffinity'] = 0
            mocap['geoms'][0]['pos'] = mocap.pop('pos')
            body = build_xml_from_dict(mocap)
            worldbody['body'].append(body['body'])
            # Add weld to equality list
            mocap['body1'] = name
            mocap['body2'] = name.replace('mocap', 'obj')
            weld = xmltodict.parse(
                # pylint: disable-next=consider-using-f-string
                """
                <weld name="{name}" body1="{body1}" body2="{body2}" solref=".02 1.5"/>
            """.format(
                    **{k: convert(v) for k, v in mocap.items()},
                ),
            )
            equality['weld'].append(weld['weld'])
        # Add geoms to XML dictionary
        for name, geom in self.geoms.items():  # pylint: disable=no-member
            assert geom['name'] == name, f'Inconsistent {name} {geom}'
            geom = geom.copy()  # don't modify original object
            for item in geom['geoms']:
                if 'contype' not in item:
                    item['contype'] = item.get('contype', 1)
                if 'conaffinity' not in item:
                    item['conaffinity'] = item.get('conaffinity', 1)
            if 'rot' in geom:
                geom['quat'] = rot2quat(geom.pop('rot'))
            body = build_xml_from_dict(geom)
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])

        # Instantiate simulator
        # print(xmltodict.unparse(self.xml, pretty=True))
        self.xml_string = xmltodict.unparse(self.xml)
        model = mujoco.MjModel.from_xml_string(self.xml_string)  # pylint: disable=no-member
        data = mujoco.MjData(model)  # pylint: disable=no-member
        
        # Store default physics parameters for clean manipulation
        self._store_default_parameters(model)
        
        # Apply randomized physics parameters
        self._apply_physics_parameters(model)
        
        # Recompute simulation intrinsics from new position
        mujoco.mj_forward(model, data)  # pylint: disable=no-member
        self.engine.update(model, data)

    def _store_default_parameters(self, model):
        """Store the default physics parameters for clean manipulation."""
        # It's crucial to store the original values so you can reset them
        # or apply multipliers to a clean slate on each reset.
        self.default_gravity = np.copy(model.opt.gravity)
        
        # Store joint-related parameters
        if hasattr(model, 'dof_damping') and model.dof_damping is not None:
            self.default_damping = np.copy(model.dof_damping)
        else:
            self.default_damping = None
            
        # Store body parameters
        if hasattr(model, 'body_mass') and model.body_mass is not None:
            self.default_mass = np.copy(model.body_mass)
        else:
            self.default_mass = None
            
        if hasattr(model, 'body_inertia') and model.body_inertia is not None:
            self.default_inertia = np.copy(model.body_inertia)
        else:
            self.default_inertia = None
            
        # Store geom parameters
        if hasattr(model, 'geom_friction') and model.geom_friction is not None:
            self.default_friction = np.copy(model.geom_friction)
        else:
            self.default_friction = None

    def _apply_physics_parameters(self, model):
        """Apply randomized physics parameters directly to the model."""
        if 'Circle' in self.task_name:
            self.one_side_param = ["damping_mult"]
        else:
            self.one_side_param = [""]
        
        # Sample new dynamics multipliers
        parameters, features_offset = self.sample_hidden_parameters(
            fix_hidden_parameters=self.fix_hidden_parameters, 
            is_out_of_distribution=self.is_out_of_distribution, 
            min_mult=self.min_mult, 
            max_mult=self.max_mult,
            out_side_param=self.one_side_param,
            # max_param_bound=max_param_bound,
            # min_param_bound=min_param_bound,
        )

        damping_mult = parameters['damping_mult']
        gravity_mult = parameters['gravity_mult']
        mass_mult = parameters['mass_mult']
        inertia_mult = parameters['inertia_mult']
        friction_mult = parameters['friction_mult']
        # Safety validations for parameter bounds
        assert 0.1 <= damping_mult <= 2.5, f"Unsafe damping multiplier: {damping_mult}"
        assert 0.1 <= gravity_mult <= 2.5, f"Unsafe gravity multiplier: {gravity_mult}"
        assert 0.1 <= mass_mult <= 2.5, f"Unsafe mass multiplier: {mass_mult}"
        assert 0.1 <= inertia_mult <= 2.5, f"Unsafe inertia multiplier: {inertia_mult}"
        assert 0.1 <= friction_mult <= 2.5, f"Unsafe friction multiplier: {friction_mult}"
        # Log all sampled parameters for debugging
        logger.info(f"All hidden parameters - damping:{damping_mult:.3f}, "
                   f"gravity:{gravity_mult:.3f}, mass:{mass_mult:.3f}, inertia:{inertia_mult:.3f}, friction:{friction_mult:.3f}")
        
        # Log if parameters are fixed
        if self.fix_hidden_parameters:
            logger.info("Parameters are FIXED (no randomization)")
        else:
            logger.info("Parameters are RANDOMIZED")
            
        # Apply gravity multiplier
        new_gravity = self.default_gravity * gravity_mult
        model.opt.gravity[:] = new_gravity
        logger.debug(f"Gravity: {self.default_gravity} -> {new_gravity}")

        hidden_parameter_features = []
        
        # Apply other parameters based on availability
        if self.default_damping is not None:
            new_damping = self.default_damping * damping_mult
            model.dof_damping[:] = new_damping
            logger.debug(f"Damping range: {new_damping.min():.3f} - {new_damping.max():.3f}")
            hidden_parameter_features.append(damping_mult - features_offset)
            
        if self.default_mass is not None:
            new_mass = self.default_mass * mass_mult
            # Only modify non-zero masses to preserve zero masses (e.g., for fixed bodies)
            mask = self.default_mass > 0
            model.body_mass[mask] = new_mass[mask]
            # Validate non-zero masses for safety
            if np.any(mask):
                assert np.all(new_mass[mask] > 0), "Non-zero mass must remain positive for stability"
                logger.debug(f"Mass range (non-zero): {new_mass[mask].min():.3f} - {new_mass[mask].max():.3f}")
                hidden_parameter_features.append(mass_mult - features_offset)
            else:
                logger.debug("No non-zero masses to modify")
        
        if self.default_inertia is not None:
            new_inertia = self.default_inertia * inertia_mult  # Inertia scales with mass
            # Only modify non-zero inertias to preserve zero inertias (e.g., for fixed bodies)
            mask = self.default_inertia > 0
            model.body_inertia[mask] = new_inertia[mask]
            # Validate non-zero inertias for stability
            if np.any(mask):
                assert np.all(new_inertia[mask] > 0), "Non-zero inertia must remain positive for stability"
            hidden_parameter_features.append(inertia_mult - features_offset)

        if self.default_friction is not None:
            new_friction = self.default_friction * friction_mult
            model.geom_friction[:] = new_friction
            # Validate friction bounds
            assert np.all(new_friction >= 0), "Friction must be non-negative"
            logger.debug(f"Friction range: {new_friction.min():.3f} - {new_friction.max():.3f}")
            hidden_parameter_features.append(friction_mult - features_offset)

        self.hidden_parameters_features = np.array(hidden_parameter_features)

    def reset_physics_parameters(self):
        """Reset physics parameters to defaults and apply new randomization."""
        if not hasattr(self, 'default_gravity'):
            raise RuntimeError("Default parameters not stored. Call build() first.")
            
        model = self.engine.model
        
        logger.info("Resetting physics parameters to defaults before applying new randomization")
        
        # Reset to defaults before applying new multipliers
        model.opt.gravity[:] = self.default_gravity
        
        if self.default_damping is not None:
            model.dof_damping[:] = self.default_damping
            
        if self.default_mass is not None:
            model.body_mass[:] = self.default_mass
            
        if self.default_inertia is not None:
            model.body_inertia[:] = self.default_inertia
            
        if self.default_friction is not None:
            model.geom_friction[:] = self.default_friction

        # Apply new randomization
        self._apply_physics_parameters(model)
        
        # Reset simulation state and perform a forward pass
        mujoco.mj_resetData(model, self.data)
        mujoco.mj_forward(model, self.data)
        
        logger.info("Physics parameter reset completed successfully")

    def rebuild(self, config=None, state=True):
        """Build a new sim from a model if the model changed."""
        if state:
            old_state = self.get_state()

        if config:
            self.parse(config)
        self.build()
        if state:
            self.set_state(old_state)
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def reset(self, build=True):
        """Reset the world. (sim is accessed through self.sim)"""
        if build:
            self.build()
        else:
            # Use clean parameter manipulation without rebuilding
            if hasattr(self, 'default_gravity'):
                self.reset_physics_parameters()
            else:
                # Fallback to full build if defaults not stored
                self.build()

    def body_com(self, name):
        """Get the center of mass of a named body in the simulator world reference frame."""
        return self.data.body(name).subtree_com.copy()

    def body_pos(self, name):
        """Get the position of a named body in the simulator world reference frame."""
        return self.data.body(name).xpos.copy()

    def body_mat(self, name):
        """Get the rotation matrix of a named body in the simulator world reference frame."""
        return self.data.body(name).xmat.copy().reshape(3, -1)

    def body_vel(self, name):
        """Get the velocity of a named body in the simulator world reference frame."""
        return get_body_xvelp(self.model, self.data, name).copy()

    def get_state(self):
        """Returns a copy of the simulator state."""
        state = {
            'time': np.copy(self.data.time),
            'qpos': np.copy(self.data.qpos),
            'qvel': np.copy(self.data.qvel),
        }
        if self.model.na == 0:
            state['act'] = None
        else:
            state['act'] = np.copy(self.data.act)

        return state

    def set_state(self, value):
        """
        Sets the state from an dict.

        Args:
        - value (dict): the desired state.
        - call_forward: optionally call sim.forward(). Called by default if
            the udd_callback is set.
        """
        self.data.time = value['time']
        self.data.qpos[:] = np.copy(value['qpos'])
        self.data.qvel[:] = np.copy(value['qvel'])
        if self.model.na != 0:
            self.data.act[:] = np.copy(value['act'])

    @property
    def model(self):
        """Access model easily."""
        return self.engine.model

    @property
    def data(self):
        """Access data easily."""
        return self.engine.data
