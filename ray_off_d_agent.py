import ray
# from deep_lfd.learning_driving.deep_learner import *
from gym_driving.envs.agents.supervised_agent import *

class RayOffAgent(SupervisedAgent):

	"""Wrapper Class for enabling ease of weight transfer"""

	def __init__(self, *args):
		super(RayOffAgent, self).__init__(*args)

	def get_weights(self):
		return self.variables.get_weights()

	def set_weights(self, weights):
		self.variables.set_weights(weights)

	def get_params(self):
		pass

	def set_params(self,params):
		pass
		
