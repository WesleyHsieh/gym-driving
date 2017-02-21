import ray
# from deep_lfd.learning_driving.deep_learner import *
from gym_driving.envs.agents.dagger_agent import *

class RayDaggerAgent(DaggerAgent):

	"""Wrapper Class for enabling ease of weight transfer"""

	def __init__(self, *args):
		super(RayDaggerAgent, self).__init__(*args)

		loss = self.learner.net.loss
		sess = self.learner.net.sess

		self.variables = ray.experimental.TensorFlowVariables(loss, sess)

	def get_weights(self):
		return self.variables.get_weights()

	def set_weights(self, weights):
		self.variables.set_weights(weights)
