import numpy as np
from gym_driving.envs.agents.agent import * 
import IPython
import copy
import time

class SupervisedAgent(Agent):
	
	def __init__(self, learner, env_list, supervisor_list):
		super(SupervisedAgent, self).__init__(learner, env_list)
		self.learner_list = [copy.deepcopy(self.learner) for _ in range(len(self.env_list))]
		self.supervisor_list = supervisor_list
		self.rewards, self.surrogate_losses = [], []

	def rollout_algorithm(self, n_trials=1):
		"""
		Rollout algorithm on underlying environment
		for one trajectory, using supervisor.
		"""
		state_list, action_list, _, _, _ = \
			self.collect_rollouts(self.env_list, self.supervisor_list)
		return state_list, action_list

	def eval_policy(self, n_trials=1):
		"""
		Evaluate underlying learner's policy. 
		"""
		state_list, action_list, reward_list, supervisor_label_list, surrogate_loss_list = \
			self.collect_rollouts(self.env_list, self.supervisor_list, self.learner_list, \
				action_mode='learner')
		self.rewards, self.surrogate_losses = reward_list, surrogate_loss_list
		return state_list, action_list, reward_list, supervisor_label_list, surrogate_loss_list

	def update_model(self, states, actions):
		"""
		Update model of underlying learner.
		"""
		self.learner.add_to_data(states, actions)
		self.learner.train_learner()
		self.learner_list = [copy.deepcopy(self.learner) for _ in range(len(self.env_list))]

	def get_statistics(self):
		"""
		Train/test accuracy, reward, surrogate loss
		"""
		train_loss, test_loss = self.learner.get_statistics()
		return train_loss, test_loss, np.mean(self.rewards), np.mean(self.surrogate_losses)

	def reset(self):
		self.learner.reset()
		self.learner_list = [copy.deepcopy(self.learner) for _ in range(len(self.env_list))]
		self.rewards, self.surrogate_losses = [], []
		
	# def __deepcopy__(self, memo):
	# 	agent = SupervisedAgent()

	def collect_rollouts(self, env_pool, supervisor_pool, learner_pool=None, seed_pool=None, action_mode='supervisor'):
		if seed_pool is None:
			seed_pool = [None for _ in range(len(env_pool))]
		if learner_pool is None:
			learner_pool = [None for _ in range(len(env_pool))]
		action_mode_pool = [action_mode for _ in range(len(env_pool))]
		params_zipped = zip(env_pool, supervisor_pool, learner_pool, seed_pool, action_mode_pool)
		pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count() - 1, len(env_pool)))
		result_lists = pool.map(collect_rollouts_wrapper, params_zipped)
		pool.close()
		pool.join()
		# result_lists = [collect_rollouts_wrapper(param) for param in params_zipped]

		state_list, action_list, reward_list, supervisor_label_list, surrogate_loss_list = zip(*result_lists)
		return state_list, action_list, reward_list, supervisor_label_list, surrogate_loss_list

def collect_rollouts_wrapper(args):
	return collect_rollouts(*args)

def collect_rollouts(env, supervisor, learner=None, seed=None, action_mode='supervisor'):
	np.random.seed(seed)
	states, actions, rewards, supervisor_labels, surrogate_losses = [], [], [], [], []
	done = False
	state = env._reset()
	while not done:
		supervisor_label = supervisor.eval_policy(env)
		action = supervisor_label
		if learner is not None:
			learner_label = learner.eval_policy(state)
			if action_mode == 'learner':
				action = learner_label
			surrogate_losses.append(action != supervisor_label)

		next_state, reward, done, info_dict = env._step(action)
		states.append(state)
		actions.append(action)
		rewards.append(reward)
		supervisor_labels.append(supervisor_label)
		state = next_state

	reward_total = sum(rewards)
	surrogate_loss_total = sum(surrogate_losses)
	return states, actions, reward_total, supervisor_labels, surrogate_loss_total
