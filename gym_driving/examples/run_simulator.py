from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *
from gym_driving.controllers.controller import *

import time
import pygame, sys
from pygame.locals import *
import random
import cProfile
import IPython
import argparse

TIMESTEPS = 1000
SLEEP_DELAY = .0001
ACC_ACTION = 5.0
STEER_ACTION = 15.0
FPS = 30
SCREEN_SIZE = (500, 500)
SCREEN_COORD = (0, 0)

CAR_X = 0
CAR_Y = 0
CAR_ANGLE = 0
VEHICLES_X = [0, 0]
VEHICLES_Y = [-100, 100]
VEHICLES_ANGLE = [0, 0]
TERRAINS = []

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# RENDER_MODE = True
CONTROLLER_MODE = 'keyboard'
RENDER_MODE = True
# CONTROLLER_MODE = 'agent'
SCREENSHOT_DIR = None
# SCREENSHOT_DIR = 'screenshots'
# os.environ["SDL_VIDEODRIVER"] = "dummy"
"""
Controller Mode:
keyboard: Up/Down to accelerate, Left/Right to steer
xbox: Left stick up/down to accelerate, right stick left/right to steer
"""

def draw_box_coords(rectangle, screen, SCREEN_COORD):
    """
    Draws corners of input rectangle on screen,
    used for debugging.

    Args:
        rectangle: rectangle object
        screen: screen object
        SCREEN_COORD: 1x2 array, coordinates of center of screen
    """
    corners = rectangle.get_corners()
    for c in corners:
        pos = (int(c[0] - SCREEN_COORD[0]), int(c[1] - SCREEN_COORD[1]))
        pygame.draw.circle(screen, 0, pos, 5, 0)
    c = rectangle.get_pos()
    pos = (int(c[0] - SCREEN_COORD[0]), int(c[1] - SCREEN_COORD[1]))
    pygame.draw.circle(screen, 0, pos, 5, 0)

def simulate_driving_agent(search_horizon=3):
    """
    Simulates one trajectory controlled by the driving search agent.

    Args:
        search_horizon: int, number of timesteps in search horizon.

    Returns:
        counter: int, number of timesteps survived in trajectory.
    """
    param_dict = {'num_cpu_cars': 5, 'main_car_starting_angles': np.linspace(-30, 30, 5), 'cpu_cars_bounding_box': [[-100.0, 1000.0], [-90.0, 90.0]]}
    pygame.init()
    fpsClock = pygame.time.Clock()
    if RENDER_MODE:
        screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Driving Simulator')
    else:
        screen = None
    simulator = DrivingEnv(render_mode=RENDER_MODE, screenshot_dir=SCREENSHOT_DIR, param_dict=param_dict)
    param_dict = {'search_horizon': search_horizon, 'driving_env': simulator}
    controller = Controller(mode='agent', param_dict=param_dict)

    done = False
    counter = 0
    simulator._reset()
    while counter < 100 and not done:
        # Checks for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        action = controller.process_input(simulator)
        # Steering only
        action = action[0]

        state, reward, done, info_dict = simulator._step(action)
        counter += 1
    return counter

    fpsClock.tick(FPS)

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


def simulate_manual_control(config_filepath=None):
    """
    Manually control the main car in the driving environment.

    Args:
        config_filepath: str, path to configuration file.
    """
    print('config_filepath', config_filepath)
    # PyGame initializations
    pygame.init()
    fpsClock = pygame.time.Clock()
    if RENDER_MODE:
        screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Driving Simulator')
    else:
        screen = None

    if config_filepath is None:
        config_filepath = '../configs/config.json'
    controller = Controller(CONTROLLER_MODE)
    simulator = DrivingEnv(render_mode=RENDER_MODE, config_filepath=config_filepath)
    states, actions, rewards = [], [], []

    time.sleep(3)

    for t in range(TIMESTEPS):
        # Checks for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        action = controller.process_input(simulator)
        # Steering only
        # action = action[0]

        state, reward, done, info_dict = simulator._step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        fpsClock.tick(FPS)

        if t == TIMESTEPS - 1:
            states.append(state)
        # time.sleep(SLEEP_DELAY)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    args = parser.parse_args()

    config_filepath = args.config
    simulate_manual_control(config_filepath)
