import numpy as np
from gym_driving.envs.agents.agent import * 
import IPython

class SupervisedAgent(Agent):
	
	def __init__(self, learner, env, supervisor):
		super(SupervisedAgent, self).__init__(learner, env)
		self.supervisor = supervisor
		self.rewards, self.surrogate_losses = [], []

	def rollout_algorithm(self):
		"""
		Rollout algorithm on underlying environment
		for one trajectory, using supervisor.
		"""
		states, actions = [], []
		done = False
		state = self.env._reset()
		while not done:
			supervisor_label = self.supervisor.rollout_policy(self.env)
			next_state, _, done, _ = self.env._step(supervisor_label)

			states.append(state)
			actions.append(supervisor_label)
			state = next_state

		return states, actions

	def eval_policy(self):
		"""
		Evaluate underlying learner's policy. 
		"""
		states, actions, rewards, supervisor_labels, surrogate_losses = [], [], [], [], []
		done = False
		state = self.env._reset()
		while not done:
			supervisor_label = self.supervisor.rollout_policy(self.env)
			action = self.learner.eval_policy(state)
			next_state, reward, done, info_dict = self.env._step(action)

			states.append(state)
			actions.append(action)
			rewards.append(reward)
			supervisor_labels.append(supervisor_label)
			surrogate_losses.append(action != supervisor_label)
			state = next_state

		self.rewards.append(sum(rewards))
		self.surrogate_losses.append(sum(surrogate_losses))

		return states, actions, self.rewards, supervisor_labels, self.surrogate_losses

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
		train_loss, test_loss = self.learner.get_statistics()
		return train_loss, test_loss, np.mean(self.rewards), np.mean(self.surrogate_losses)

	def reset(self):
		self.learner.reset()
		self.rewards, self.surrogate_losses = [], []
		