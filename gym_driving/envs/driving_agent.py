from gym_driving.envs.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.envs.terrain import *

from copy import deepcopy
import heapq

class DrivingAgent():
	"""
	Agent that autonomously drives a car.
	"""
	def __init__(self):
		pass

	def simulate_actions(self, actions, env):
		new_env = deepcopy(env)
		states, rewards, dones, info_dicts, = [], [], [], []
		for action in actions:
			state, reward, done, info_dict = new_env._step(action)
			states.append(state)
			rewards.append(reward)
			dones.append(done)
			info_dicts.append(info_dict)
		return new_env, states, rewards, dones, info_dicts

	def successor_func(self, action_list, env, num_steps=5):
		actions = [0, 1, 2]
		successors = []
		for action in actions:
			new_action_list = list(action_list).append(action)
			new_env, states, _, _, info_dicts = self.simulate_actions(action, env)
			new_cost = self.cost_func(states[0], info_dicts[0])
			new_done = len(new_action_list) == num_steps
			successors.append([new_action_list, new_env, new_cost, new_done])
		return successors

	def cost_func(self, state, info_dict):
		# TODO: Cost function
		return 0

	def search_agent(self, starting_state, num_steps=5):
		# Actions map to environments
		visited = set()
		# List of tuples of (cost, action_list, env, done)
		fringe = []
		heapq.heappush(fringe, (0, (), starting_state, False))
		while len(fringe) > 0:
			cost, action_list, env, done = heapq.heappop(fringe)
			if done:
				return cost, action_list, env
			elif action_list not in visited:
				visited.add(action_list)
				# Evaluates all successors, returns list of (action_list, env) tuples
				successors = self.successor_func(action_list, env, num_steps)
				for new_action_list, new_env, new_cost, new_done in successors:
					new_tuple = (cost + new_cost, new_action_list, new_env, new_done)
					heapq.heappush(fringe, new_tuple)

	def driving_agent(self, driving_env, num_steps=5, step_size=1):
		cost, action_list, new_env = self.search_agent(driving_env, num_steps)
		return action_list[0]
