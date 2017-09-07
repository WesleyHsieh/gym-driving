from gym_driving.envs.environment import *
from gym_driving.assets.car import *
from gym_driving.assets.terrain import *

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import IPython
import pickle
import json
import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
logger = logging.getLogger(__name__)

class DrivingEnv(gym.Env):
    """
    Wrapper class for driving simulator that
    implements the OpenAI Gym interface.
    """
    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second' : 50
    # }
    def __init__(self, render_mode=True, screen=None, config_filepath=None):
        """
        Initializes driving environment interface,
        passes most arguments down to underlying environment.

        Args:
            render_mode: boolean, whether to render.
            screen: PyGame screen object, used for rendering.
                Creates own screen object if existing one is not passed in.
            config_filepath: str, path to configuration file.
        """

        if config_filepath is None:
            base_dir = os.path.dirname(__file__)
            config_filepath = os.path.join(base_dir, '../configs/config.json')
        param_dict = json.load(open(config_filepath, 'r'))

        self.num_cpu_cars = param_dict['num_cpu_cars']
        self.main_car_starting_angles = param_dict['main_car_starting_angles']
        self.cpu_cars_bounding_box = param_dict['cpu_cars_bounding_box']
        self.screen_size = param_dict['screen_size']
        self.logging_dir = param_dict['logging_dir']
        self.logging_rate = param_dict['logging_rate']
        self.time_horizon = param_dict['time_horizon']
        self.terrain_params = param_dict['terrain_params']
        self.state_space = param_dict['state_space']
        self.control_space = param_dict['control_space']
        self.param_dict = param_dict

        # Default options for PyGame screen, terrain
        if screen is None:
            screen = pygame.display.set_mode(self.screen_size)
            # pygame.display.set_caption('Driving Simulator')

        if self.logging_dir is not None and not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)
        self.screen = screen
        self.environment = Environment(render_mode=render_mode, screen_size=self.screen_size, \
                screen=self.screen, param_dict=self.param_dict)
        self.render_mode = render_mode
        low, high, step = param_dict['steer_action']
        if self.control_space == 'discrete':
            # 0, 1, 2 = Steer left, center, right
            action_space = np.linspace(low, high, step)
            self.action_space = spaces.Discrete(len(action_space))
        elif self.control_space == 'continuous':
            self.action_space = spaces.Box(low=low, high=high, shape=(1,))

        # Limits on x, y, angle
        if self.state_space == 'positions':
            low = np.tile(np.array([-10000.0, -10000.0]), self.num_cpu_cars)
            high = np.tile(np.array([10000.0, 10000.0]), self.num_cpu_cars)
            low = np.concatenate([low, [-10000.0, -10000.0, 0.0]])
            high = np.concatenate([high, [10000.0, 10000.0, 360.0]])
            self.observation_space = spaces.Box(low, high)
        elif self.state_space == 'image':
            w, h = param_dict['screen_size']
            self.observation_space = spaces.Box(low=0, high=255, shape=(w, h))
        self.exp_count = self.iter_count = 0

        # self._seed()
        # self.reset()
        # self.viewer = None

        # self.steps_beyond_done = None

        # # Just need to initialize the relevant attributes
        # self._configure()

    # def _configure(self, display=None):
    #     self.display = display

    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def _render(self, mode='human', close=False):
        """
        Dummy render command for gym interface.

        Args:
            mode: str
            close: boolean
        """
        pass

    def render(self, mode='human', close=False):
        """
        Dummy render command for gym interface.

        Args:
            mode: str
            close: boolean
        """
        pass

    def _step(self, action):
        """
        Updates the environment for one step.

        Args:
            action: 1x2 array, steering / acceleration action.

        Returns:
            state: array, state of environment.
                Can be positions and angles of cars, or image of environment
                depending on configuration.
            reward: float, reward from action taken.
            done: boolean, whether trajectory is finished.
            info_dict: dict, contains information about environment that may
                not be included in the state.
        """
        self.iter_count += 1
        state, reward, done, info_dict = self.environment.step(action)
        if self.logging_dir is not None and self.iter_count % self.logging_rate == 0:
            self.log_state(state)
        if self.iter_count >= self.time_horizon:
            done = True
        return state, reward, done, {}

    def _reset(self):
        """
        Resets the environment.

        Returns:
            state: array, state of environment.
        """
        self.exp_count += 1
        self.iter_count = 0
        self.screen = pygame.display.set_mode(self.screen_size)
        state = self.environment.reset(self.screen)
        return state

    def log_state(self, state):
        """
        Logs the current step.

        Args:
            state: array, state of environment.
        """
        if self.state_space == 'positions':
            file_name = 'log.txt'
            with open(file_name, 'a') as outfile:
                outfile.write(state)
        elif self.state_space == 'image':
            image_name = 'exp_{}_iter_{}.png'.format(self.exp_count, self.iter_count)
            image_path = os.path.join(self.logging_dir, image_name)
            pygame.image.save(self.screen, image_path)

    def simulate_actions(self, actions, noise=0.0, state=None):
        """
        Simulate a sequence of actions.

        Args:
            noise: float, standard deviation of zero-mean Gaussian noise
            state: dict, internal starting state of environment.
                Currently set as the positions, velocities, and angles of
                all cars.

        Returns:
            states: list, list of states in trajectory.
            rewards: list, list of rewards in trajectory.
            dones: list, list of dones in trajectory.
            info_dicts: list, list of info dicts in trajectory.
        """
        states, rewards, dones, info_dicts = self.environment.simulate_actions(actions, noise, state)
        return states, rewards, dones, info_dicts

    def __deepcopy__(self, memo):
        """
        Deep copy envrionemnt.
        """
        env = DrivingEnv(render_mode=self.render_mode, \
            screen_size=self.screen_size, screen=None, terrain=None, \
            logging_dir=self.logging_dir, logging_rate=self.logging_rate, \
            param_dict=self.param_dict)
        return env
