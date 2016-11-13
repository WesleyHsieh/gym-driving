from environment import *
from car import *

class Simulator:
    """
    Generic wrapper class for simulating
    environments.
    TODO: Make compatible with OpenAI Gym interface.
    """
    def __init__(self):
        self.environment = Environment()

    def get_state(self):
        return self.environment.get_state()

    def take_action(self, action):
        reward = self.environment.take_action(action)
        return reward

