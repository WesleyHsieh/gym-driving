import numpy as np
from gym_driving.envs.agents.supervised_agent import * 
import IPython

class DaggerAgent(SupervisedAgent):
	def __init__(self, learner, env, supervisor):
		super(DaggerAgent, self).__init__(learner, env, supervisor)
		self.state_list, self.action_list = None, None

	def rollout_algorithm(self, n_trials=1):
		"""
		Rollout algorithm on underlying environment
		for one trajectory, using supervisor.
		"""
		# Use rollouts from previous policy evaluation except for first iteration
		if self.state_list is None and self.action_list is None:
			return super(DaggerAgent, self).rollout_algorithm(n_trials)
		else:
			return self.state_list, self.action_list

	def eval_policy(self, n_trials=1):
		"""
		Evaluate underlying learner's policy. 
		"""
		state_list, action_list, reward_list, supervisor_label_list, surrogate_loss_list = \
			self.collect_rollouts(self.env_list, self.supervisor_list, self.learner_list, \
				action_mode='learner')
		self.state_list, self.action_list = state_list, action_list
		self.rewards, self.surrogate_losses = reward_list, surrogate_loss_list
		return state_list, action_list, reward_list, supervisor_label_list, surrogate_loss_list