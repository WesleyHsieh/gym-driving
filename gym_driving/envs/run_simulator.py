from gym_driving.envs.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.envs.terrain import *
from gym_driving.envs.controller import *

import time
import pygame, sys
from pygame.locals import *
import random
import cProfile

TIMESTEPS = 100
SLEEP_DELAY = .05
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
VEHICLES_ANGLE= [0, 0]
TERRAINS = []

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# GRAPHICS_MODE = True
# CONTROLLER_MODE = 'keyboard'
GRAPHICS_MODE = False #True
CONTROLLER_MODE = 'agent'
SCREENSHOT_DIR = None
# SCREENSHOT_DIR = 'screenshots'
"""
Controller Mode:
keyboard: Up/Down to accelerate, Left/Right to steer
xbox: Left stick up/down to accelerate, right stick left/right to steer
"""

def draw_box_coords(rectangle, screen, SCREEN_COORD):
    corners = rectangle.get_corners()
    for c in corners:
        pos = (int(c[0] - SCREEN_COORD[0]), int(c[1] - SCREEN_COORD[1]))
        pygame.draw.circle(screen, 0, pos, 5, 0)
    c = rectangle.get_pos()
    pos = (int(c[0] - SCREEN_COORD[0]), int(c[1] - SCREEN_COORD[1]))
    pygame.draw.circle(screen, 0, pos, 5, 0)

def simulate_driving_agent(search_horizon=3):
    pygame.init()
    fpsClock = pygame.time.Clock()
    if GRAPHICS_MODE:
        screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Driving Simulator')
    else:
        screen = None
    simulator = DrivingEnv(graphics_mode=GRAPHICS_MODE, screenshot_dir=SCREENSHOT_DIR)
    param_dict = {'search_horizon': search_horizon, 'driving_env': simulator}
    controller = Controller(mode='agent', param_dict=param_dict)
    
    done = False
    counter = 0
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


def simulate_manual_control():
    # PyGame initializations
    pygame.init()
    fpsClock = pygame.time.Clock()
    if GRAPHICS_MODE:
        screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Driving Simulator')
    else:
        screen = None

    # Add the terrain
    # for i in random.sample(xrange(0, 32), 12):
    #     TERRAINS.append(Terrain(-2048 + i*128, -128 + (i%2) * 128, 128, 128, 'icegrass', screen, SCREEN_SIZE))
    # for i in random.sample(xrange(0, 32), 8):
    #     TERRAINS.append(Terrain(-2048 + i*128, 128 + (i%2) * 128, 128, 128, 'ice', screen, SCREEN_SIZE))
    # for i in random.sample(xrange(0, 32), 12):
    #     TERRAINS.append(Terrain(-2048 + i*128, 384 + (i%2) * 128, 128, 128, 'icegrass', screen, SCREEN_SIZE))

    # TERRAINS.append(Terrain(0, -2000, 20000, 3900, 'grass', screen, SCREEN_SIZE))
    # TERRAINS.append(Terrain(0, 0, 20000, 100, 'road', screen, SCREEN_SIZE))
    # TERRAINS.append(Terrain(0, 2000, 20000, 3900, 'grass', screen, SCREEN_SIZE))

    controller = Controller(CONTROLLER_MODE)
    simulator = DrivingEnv(graphics_mode=GRAPHICS_MODE, screenshot_dir=SCREENSHOT_DIR)
    #simulator = DrivingEnv(GRAPHICS_MODE, SCREEN_SIZE, screen, TERRAINS)
    states, actions, rewards = [], [], []

    for t in range(TIMESTEPS):
        # Checks for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        action = controller.process_input(simulator)
        # Steering only
        action = action[0]

        state, reward, done, info_dict = simulator._step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        # print("State, Action, Next State")
        # print(states[t-1])
        # print(action)
        # print(states[t])

        fpsClock.tick(FPS)

        if t == TIMESTEPS - 1:
            states.append(state)
        time.sleep(SLEEP_DELAY)

if __name__ == '__main__':
    # start = time.time()

    # counts = np.array([simulate_driving_agent() for _ in range(10)])
    # mean_time_survived = np.mean(counts)
    # print("Mean Time Survived", mean_time_survived)

    # simulate_manual_control()

    # end = time.time()
    # print("Time Elapsed: ", end - start)

    # cProfile.run('run_driving_agent_experiment()')
    run_driving_agent_experiment()

    # pygame.init()
    # fpsClock = pygame.time.Clock()
    # if GRAPHICS_MODE:
    #     screen = pygame.display.set_mode(SCREEN_SIZE)
    #     pygame.display.set_caption('Driving Simulator')
    # else:
    #     screen = None
    # #controller = Controller(mode='agent', param_dict=param_dict)
    # simulator = DrivingEnv(graphics_mode=GRAPHICS_MODE, screenshot_dir=SCREENSHOT_DIR)

    # # Test
    # start = time.time()
    # action = 0
    # for _ in range(3125):
    #     simulator._step(action)
    # end = time.time()
    # print("Time Elapsed: ", end - start)
            

# Batch data by trajectory
# Batches of image data, labels