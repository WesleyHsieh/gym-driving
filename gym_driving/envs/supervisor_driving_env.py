from gym_driving.envs.environment import *
from gym_driving.envs.car import *
from gym_driving.envs.terrain import *
from gym_driving.envs.driving_env import *
from gym_driving.envs.controller import *
from gym_driving.envs.agents.driving_agent import *

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import IPython
from copy import deepcopy

logger = logging.getLogger(__name__)

class SupervisorDrivingEnv(gym.Env):
    def __init__(self, graphics_mode=False, screen_size=(512, 512), screen=None, terrain=None, screenshot_dir=None, screenshot_rate=10, time_horizon=100, param_dict=None):
        self.environment = DrivingEnv(graphics_mode, screen_size, screen, terrain, screenshot_dir, screenshot_rate, time_horizon, param_dict)
        self.supervisor = DrivingAgent()
        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space

    def _step(self, action):
        supervisor_label = self.supervisor.eval_policy(self.environment, None)
        state, _, done, info_dict = self.environment._step(action)
        reward = supervisor_label == action
        return state, reward, done, info_dict
        
    def _reset(self):
        return self.environment._reset()
        
    def _render(self, mode='human', close=False):
        return self.environment._render(mode, close)

    def save_image(self):
        self.driving_env.save_image()

    def simulate_actions(self, actions, noise=0.0, state=None):
        return self.environment.simulate_actions(actions, noise, state)

    def __deepcopy__(self, memo):
        supervisor_driving_env = SupervisorDrivingEnv(graphics_mode=self.graphics_mode, \
            screen_size=self.screen_size, screen=None, terrain=None, \
            screenshot_dir=self.screenshot_dir, screenshot_rate=self.screenshot_rate, \
            param_dict=self.param_dict)
        supervisor_driving_env.environment = copy.deepcopy(self.environment)
        supervisor_driving_env.supervisor = copy.deepcopy(self.supervisor)
        return supervisor_driving_env

    def _render(self, mode='human', close=False):
        pass