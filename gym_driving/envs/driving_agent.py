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

	def successor_func(self, action_list, env, num_steps=5):
		actions = [0, 1, 2]
		successors = []
		for action in actions:
			new_action_list = list(action_list) + [action]
			states, _, _, info_dicts = env.simulate_actions(new_action_list)
			new_env = env
			new_cost = self.cost_heuristic_func(states[-1], info_dicts[-1])
			# print("Action Tried: ", new_action_list)
			# print("New Cost: ", new_cost)
			new_done = len(new_action_list) == num_steps
			successors.append([new_action_list, new_env, new_cost, new_done])
		return successors

	def cost_heuristic_func(self, state, info_dict):
		terrain_collisions = info_dict['terrain_collisions']
		car_collisions = info_dict['car_collisions']
		angle_diff = info_dict['angle_diff']
		crash = 99999 * \
			(any([t.texture == 'grass' for t in terrain_collisions]) or len(car_collisions) >= 1)
		cost = crash + angle_diff
		return cost

	def search_agent(self, starting_state, num_steps=7):
		# Actions map to environments
		visited = set()
		# List of tuples of (cost, action_list, env, done)
		fringe = []
		heapq.heappush(fringe, (0, (), starting_state, False))
		while len(fringe) > 0:
			cost, action_list, env, done = heapq.heappop(fringe)
			action_list = tuple(action_list)
			if done:
				return cost, action_list, env
			elif action_list not in visited:
				print("Action List", action_list)
				self.counter += 1
				visited.add(action_list)
				# Evaluates all successors, returns list of (action_list, env) tuples
				successors = self.successor_func(action_list, env, num_steps)
				for new_action_list, new_env, new_cost, new_done in successors:
					new_tuple = (cost + new_cost, new_action_list, new_env, new_done)
					heapq.heappush(fringe, new_tuple)

	def driving_agent(self, driving_env, num_steps=7, step_size=1):
		self.counter = 0
		cost, action_list, new_env = self.search_agent(driving_env, num_steps)
		print("Length of actions", len(action_list))
		print("Cost of actions", cost)
		print("Number of nodes expanded", self.counter)
		self.counter = 0
		return action_list[0]