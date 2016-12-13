from environment import *
from car import *
from terrain import *

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class DrivingEnv(gym.Env):
    """
    Generic wrapper class for simulating
    environments.
    """
    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second' : 50
    # }

    def __init__(self, graphics_mode=True, screen_size=(500, 500), screen=None, terrain=None):
        # Default options for PyGame screen, terrain
        if screen is None:
            screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption('Driving Simulator')
        if terrain is None:
            terrain = []
            terrain.append(Terrain(0, -2000, 20000, 3900, 'grass', screen, screen_size))
            terrain.append(Terrain(0, 0, 20000, 100, 'road', screen, screen_size))
            terrain.append(Terrain(0, 2000, 20000, 3900, 'grass', screen, screen_size))
        self.environment = Environment(graphics_mode, screen_size, screen, terrain)

        # 0, 1, 2 = Steer left, center, right
        self.action_space = spaces.Discrete(2)
        # Limits on x, y, angle
        low = np.array([-10000.0, -10000.0, 0.0])
        high = np.array([10000.0, 10000.0, 360.0])
        self.observation_space = spaces.Box(low, high)

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

    def _step(self, action):
        state, reward, done, info_dict = self.environment.take_action(action)
        # print(state, reward, done, info_dict)
        return state, reward, done, info_dict
        
    def _reset(self):
        state = self.environment.reset()
        return state

    def _render(self, mode='human', close=False):
        return None
