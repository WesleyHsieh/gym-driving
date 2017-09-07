import ray
from gym_driving.agents.dart_agent import *

class RayDartAgent(DartAgent):

    """Wrapper Class for enabling ease of weight transfer"""

    def __init__(self, *args):
        super(RayDartAgent, self).__init__(*args)

    def get_weights(self):
        return self.learner.get_weights()

    def set_weights(self, weights):
        self.learner.set_weights(weights)

    def get_params(self):
        pass

    def set_params(self, params):
        self.eps = params[0]
