import numpy as np

class Agent(object):
	def __init__(self, learner, env):
		self.learner = learner
		self.env = env

	def rollout_algorithm(self):
		"""
		Rollout algorithm on underlying environment
		for one trajectory.
		"""
		states, actions = self.learner.rollout_algorithm(env)

	def eval_policy(self):
		"""
		Evaluate underlying learner's policy. 
		"""
		output_label = self.learner.eval_policy(img)
		return output_label

	def update_model(self, states, actions):
		"""
		Update model of underlying learner.
		"""
		self.learner.add_to_data(states, actions)
		self.learner.train_learner()

	def get_statistics(self):
		"""
		Train/test accuracy, reward, surrogate loss
		"""
		pass


