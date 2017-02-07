from gym_driving.envs.environment import *
from gym_driving.envs.car import *
from gym_driving.envs.terrain import *

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import IPython

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

    def __init__(self, graphics_mode=False, screen_size=(512, 512), screen=None, terrain=None, screenshot_dir=None, screenshot_rate=10, num_cpu_cars=10, time_horizon=100):
        # Default options for PyGame screen, terrain
        if screen is None:
            screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption('Driving Simulator')
        self.screenshot_dir = screenshot_dir
        if screenshot_dir is not None and not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
        if terrain is None:
            terrain = []
            terrain.append(Terrain(x=0, y=-2000, width=20000, length=3800, texture='grass', \
                screen=screen, screen_size=screen_size, graphics_mode=graphics_mode))
            terrain.append(Terrain(x=0, y=0, width=20000, length=200, texture='road', \
                screen=screen, screen_size=screen_size, graphics_mode=graphics_mode))
            terrain.append(Terrain(x=0, y=2000, width=20000, length=3800, texture='grass', \
                screen=screen, screen_size=screen_size, graphics_mode=graphics_mode))
        self.screen = screen
        self.environment = Environment(graphics_mode=graphics_mode, screen_size=screen_size, \
                screen=screen, terrain=terrain, num_cpu_cars=num_cpu_cars)
        self.graphics_mode = graphics_mode

        # 0, 1, 2 = Steer left, center, right
        self.action_space = spaces.Discrete(2)

        # Limits on x, y, angle
        low = np.tile(np.array([-10000.0, -10000.0, 0.0]), num_cpu_cars + 1)
        high = np.tile(np.array([10000.0, 10000.0, 360.0]), num_cpu_cars + 1)
        self.observation_space = spaces.Box(low, high)

        self.exp_count = self.iter_count = 0
        self.screenshot_rate = screenshot_rate
        self.time_horizon = time_horizon
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
        self.iter_count += 1
        action = np.array([action, 1.0])
        state, reward, done, info_dict = self.environment.take_action(action)
        # print(state, reward, done, info_dict)
        if self.screenshot_dir is not None and self.iter_count % self.screenshot_rate == 0:
            self.save_image()
        if self.iter_count >= self.time_horizon:
            done = True
        state = pygame.surfarray.array2d(self.screen).astype(np.uint8)
        return state, reward, done, info_dict
        
    def _reset(self):
        self.exp_count += 1
        self.iter_count = 0
        state = self.environment.reset()
        state = pygame.surfarray.array2d(self.screen).astype(np.uint8)
        return state

    def _render(self, mode='human', close=False):
        return None

    def save_image(self):
        image_name = 'exp_{}_iter_{}.png'.format(self.exp_count, self.iter_count)
        image_path = os.path.join(self.screenshot_dir, image_name)
        pygame.image.save(self.screen, image_path)

    def simulate_actions(self, actions, noise=0.0, state=None):
        return self.environment.simulate_actions(actions, noise, state)