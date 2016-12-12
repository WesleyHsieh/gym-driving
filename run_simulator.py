from car import *
from environment import *
from simulator import *
from terrain import *
from controller import *

import time
import pygame, sys
from pygame.locals import *
import random

TIMESTEPS = 500
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
GRAPHICS_MODE = True
CONTROLLER_MODE = 'keyboard'
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

if __name__ == '__main__':
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

    TERRAINS.append(Terrain(0, -2000, 20000, 3900, 'grass', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(0, 0, 20000, 100, 'road', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(0, 2000, 20000, 3900, 'grass', screen, SCREEN_SIZE))

    controller = Controller(ACC_ACTION, STEER_ACTION, CONTROLLER_MODE)
    simulator = DrivingEnv()
    #simulator = DrivingEnv(GRAPHICS_MODE, SCREEN_SIZE, screen, TERRAINS)
    states, actions, rewards = [], [], []

    for t in range(TIMESTEPS):
        # Checks for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        action = controller.process_input()

        state, reward, done, info_dict = simulator._step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        # print "State, Action, Next State"
        # print states[t-1]
        # print action
        # print states[t]

        fpsClock.tick(FPS)

        if t == TIMESTEPS - 1:
            states.append(state)
        time.sleep(SLEEP_DELAY)
