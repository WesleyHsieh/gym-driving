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

def simulate_manual_control(config_filepath=None):
    """
    Manually control the main car in the driving environment.

    Args:
        config_filepath: str, path to configuration file.
    """
    # PyGame initializations
    pygame.init()
    fpsClock = pygame.time.Clock()
    if RENDER_MODE:
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
