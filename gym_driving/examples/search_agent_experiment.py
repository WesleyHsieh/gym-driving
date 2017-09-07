import time
import pygame, sys
from pygame.locals import *
import IPython
import random
import numpy as np

from gym_driving.envs.driving_env import *
from gym_driving.agents.search_agent import *

render_mode = True
config_filepath = '../configs/driving_experiment_config.json'

def simulate_driving_agent(search_horizon=3):
    """
    Simulates one trajectory controlled by the driving search agent.

    Args:
        search_horizon: int, number of timesteps in search horizon.

    Returns:
        counter: int, number of timesteps survived in trajectory.
    """
    pygame.init()
    screen = None
    env = DrivingEnv(render_mode=render_mode, screen=screen, config_filepath=config_filepath)

    agent_param_dict = {'search_horizon': search_horizon}
    agent = SearchAgent(param_dict=agent_param_dict, env=env)

    done = False
    counter = 0
    env._reset()
    while counter < 100 and not done:
        # Checks for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        action = agent.eval_policy(env)
        for curr_action in action:
            state, reward, done, info_dict = env._step(curr_action)
            counter += 1
    return counter

def run_driving_agent_experiment(num_experiments=50):
    """
    Simulates multiple trajectories controlled by the driving search agent.

    Args:
        num_experiments: Number of trajectories to run.
    """
    # search_horizons = [3, 5, 7]
    search_horizons = [5]
    result_dict = {}
    for search_horizon in search_horizons:
        print("Running Search Horizon: {}".format(search_horizon))
        scores, times = [], []
        param_dict = {'search_horizon': search_horizon}
        for _ in range(num_experiments):
            start = time.time()
            scores.append(simulate_driving_agent(search_horizon))
            end = time.time()
            times.append(end - start)
        result_dict[search_horizon] = {'mean_score': np.mean(np.array(scores)), \
            'mean_time': np.mean(np.array(times))}
        print("Results for search horizon = {}: ".format(search_horizon))
        print("Scores: ", scores)
        print("Times: ", times)
        print(result_dict[search_horizon])

    for search_horizon in search_horizons:
        print("Results for search horizon = {}: ".format(search_horizon))
        print(result_dict[search_horizon])

if __name__ == '__main__':
    run_driving_agent_experiment()
