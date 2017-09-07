from gym_driving.envs.environment import *
from gym_driving.assets.car import *
from gym_driving.assets.terrain import *
from gym_driving.envs.driving_env import *
from gym_driving.controllers.controller import *
# from gym_driving.agents.driving_agent import *

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
    """
    Wrapper class for driving simulator that
    implements the OpenAI Gym interface.

    In particular, takes in a baseline driving environment,
    and returns rewards of booleans whether
    the action taken matches the supervisor's label.

    The input supervisor must implement eval_policy(env, state),
    which returns the supervisor's labeled action for the current
    time step.
    """
    def __init__(self, supervisor, render_mode=True, screen=None, config_filepath=None):
        """
        Initializes driving environment interface,
        passes most arguments down to underlying environment.

        Args:
            supervisor: supervisor object, used for querying rewards.
            render_mode: boolean, whether to render.
            screen: PyGame screen object, used for rendering.
                Creates own screen object if existing one is not passed in.
            config_filepath: str, path to configuration file.
        """
        self.environment = DrivingEnv(render_mode, screen, config_filepath)
        self.supervisor = supervisor
        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space

    def _step(self, action):
        """
        Updates the environment for one step.
        Reward is now whether the supervisor's label matches
        the action taken by the agent.

        Args:
            action: 1x2 array, steering / acceleration action.

        Returns:
            state: array, state of environment.
                Can be positions and angles of cars, or image of environment
                depending on configuration.
            reward: float, reward from action taken.
                Currently set to whether the action taken matches the supervisor's label.
            done: boolean, whether trajectory is finished.
            info_dict: dict, contains information about environment that may
                not be included in the state.
        """
        supervisor_label = self.supervisor.eval_policy(self.environment, None)
        state, _, done, info_dict = self.environment._step(action)
        reward = supervisor_label == action
        return state, reward, done, info_dict

    def _reset(self):
        """
        Resets the environment.

        Returns:
            state: array, state of environment.
        """
        return self.environment._reset()

    def _render(self, mode='human', close=False):
        """
        Dummy render command for gym interface.

        Args:
            mode: str
            close: boolean
        """
        return self.environment._render(mode, close)

    def save_image(self):
        """
        Saves current image of environment.
        """
        self.driving_env.save_image()

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
        return self.environment.simulate_actions(actions, noise, state)

    def __deepcopy__(self, memo):
        """
        Deep copy envrionemnt.
        """
        supervisor_driving_env = SupervisorDrivingEnv(graphics_mode=self.graphics_mode, \
            screen_size=self.screen_size, screen=None, terrain=None, \
            screenshot_dir=self.screenshot_dir, screenshot_rate=self.screenshot_rate, \
            param_dict=self.param_dict)
        supervisor_driving_env.environment = copy.deepcopy(self.environment)
        supervisor_driving_env.supervisor = copy.deepcopy(self.supervisor)
        return supervisor_driving_env
