from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

from copy import deepcopy
import heapq

class SearchAgent():
    """
    Agent that autonomously drives a car.
    """
    def __init__(self, param_dict=None, env=None):
        if param_dict is None:
            self.search_horizon = 5
        else:
            self.search_horizon = param_dict['search_horizon']
        self.actions = [0, 1, 2, 3, 4]
        # self.actions = [0, 1, 2]
        self.reset()

    def eval_policy(self, env, search_horizon=5):
        self.driving_env = env
        self.search_horizon = search_horizon
        self.counter = 0
        cost, curr_heuristic_cost, action_list = self.search_agent()
        # print("Length of actions", len(action_list))
        # print("Final Action List", action_list)
        # print("Cost of actions", cost)
        # print("Number of nodes expanded", self.counter)
        self.counter = 0
        self.previous_path = tuple(action_list[1:])
        self.curr_heuristic_cost = curr_heuristic_cost
        # print('cost', cost)
        return action_list[0:1]

    def search_agent(self):
        # List of tuples of (total_cost, curr_heuristic_cost, action_list, simulator_state, done)
        visited = set()
        fringe = []
        heapq.heappush(fringe, (100.0, 100.0, (), None, False))
        # Warm start
        if len(self.previous_path) > 0:
            heapq.heappush(fringe, self.simulate_actions(self.previous_path,
                                                         simulator_state=None,
                                                         recompute_all=True))
        while len(fringe) > 0:
            cost, curr_heuristic_cost, action_list, simulator_state, done = heapq.heappop(fringe)
            action_list = tuple(action_list)
            if done:
                return cost, curr_heuristic_cost, action_list
            elif action_list not in visited:
                # print("Action List", action_list)
                self.counter += 1
                visited.add(action_list)
                # Evaluates all successors, returns list of (action_list, env) tuples
                successors = self.successor_func(action_list, simulator_state)
                for new_cost, new_heuristic_cost, new_action_list, new_simulator_state, new_done in successors:
                    new_total_cost = cost + new_cost - curr_heuristic_cost + new_heuristic_cost
                    new_tuple = (new_total_cost, new_heuristic_cost, new_action_list, new_simulator_state, new_done)
                    heapq.heappush(fringe, new_tuple)

    def successor_func(self, action_list, simulator_state=None):
        successors = []
        for action in self.actions:
            new_action_list = list(action_list) + [action]
            successor = self.simulate_actions(new_action_list, simulator_state)
            successors.append(successor)
        return successors

    def simulate_actions(self, action_list, simulator_state=None, recompute_all=False):
        # print("action_list", action_list)
        if recompute_all:
            states, _, dones, info_dicts = self.driving_env.simulate_actions(action_list, \
                                                                             noise=0.0,
                                                                             state=simulator_state)
            cost = sum([self.cost_func(states[i], info_dicts[i]) for i in range(len(states))])
        else:
            states, _, dones, info_dicts = self.driving_env.simulate_actions(action_list[-1:],
                                                                             noise=0.0,
                                                                             state=simulator_state)
            cost = self.cost_func(states[-1], info_dicts[-1])
        heuristic_cost = self.heuristic_func(states[-1], info_dicts[-1])
        simulator_state = info_dicts[-1]['compact_state']
        done = len(action_list) == self.search_horizon or any(dones)
        output_tuple = (cost, heuristic_cost, action_list, simulator_state, done)
        return output_tuple

    def cost_func(self, state, info_dict):
        terrain_collisions = info_dict['terrain_collisions']
        car_collisions = info_dict['car_collisions']
        crash_cost = 1e10 * \
                     (any([t.texture == 'grass' for t in terrain_collisions]) or len(car_collisions) >= 1)
        time_cost = 1.0
        # car_distance_cost = info_dict['distance_to_nearest_car']
        cost = crash_cost + time_cost #+ car_distance_cost
        return cost

    def heuristic_func(self, state, info_dict):
        end_pos, max_speed = 2000.0, 20.0
        curr_pos = info_dict['main_car']['x']
        # car_distance_cost = max(20 - info_dict['distance_to_nearest_car'], 0)
        time_cost = (end_pos - curr_pos) / max_speed
        return time_cost #+ car_distance_cost

    def reset(self):
        self.previous_path = ()
        self.curr_heuristic_cost = 100.0
