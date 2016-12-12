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

    TERRAINS.append(Terrain(0, -2000, 20000, 3700, 'grass', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(0, 0, 20000, 300, 'road', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(0, 2000, 20000, 3700, 'grass', screen, SCREEN_SIZE))

    if GRAPHICS_MODE:
        for t in TERRAINS:
            t.update_graphics(SCREEN_COORD)

        # Add the car
        main_car_image = pygame.image.load('images/main_car_lite.png')
        screen.blit(main_car_image, (CAR_X, CAR_Y))

        # Add other vehicles
        red_car_image = pygame.image.load('images/red_car_lite.png')
        screen.blit(red_car_image, (VEHICLES_X[0], VEHICLES_Y[0]))
        orange_car_image = pygame.image.load('images/orange_car_lite.png')
        screen.blit(orange_car_image, (VEHICLES_X[1], VEHICLES_Y[1]))

    controller = Controller(ACC_ACTION, STEER_ACTION, CONTROLLER_MODE)
    simulator = DrivingEnv(SCREEN_SIZE, TERRAINS)
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

        # Update Car Location
        CAR_X = state['main_car']['x']
        CAR_Y = state['main_car']['y']
        CAR_ANGLE = state['main_car']['angle']

        # Update Vehicle Locations
        for i in range(len(state['other_cars'])):
            VEHICLES_X[i] = state['other_cars'][i]['x']
            VEHICLES_Y[i] = state['other_cars'][i]['y']
            VEHICLES_ANGLE[i] = state['other_cars'][i]['angle']

        if GRAPHICS_MODE:
            # Clear the screen
            screen.fill(WHITE)
            main_car_image_update = pygame.transform.rotate(main_car_image, -CAR_ANGLE)
            red_car_image_update = pygame.transform.rotate(red_car_image, -CAR_ANGLE)
            orange_car_image_update = pygame.transform.rotate(orange_car_image, -CAR_ANGLE)

            # Update SCREEN_COORD
            SCREEN_COORD = (CAR_X - SCREEN_SIZE[0]/2, CAR_Y - SCREEN_SIZE[1]/2)

            # Update terrain graphics
            for t in TERRAINS:
                t.update_graphics(SCREEN_COORD)
                draw_box_coords(t, screen, SCREEN_COORD)

            # Update car graphics
            car = simulator.environment.main_car
            corners, center, angle = car.get_corners(), car.get_pos(), car.angle
            x_offset = (np.abs((car.width - car.length) * np.cos(np.radians(angle))) + car.length) / 2
            y_offset = (np.abs((car.width - car.length) * np.sin(np.radians(angle))) + car.length) / 2
            new_pos = (int(SCREEN_SIZE[0] / 2 - x_offset), int(SCREEN_SIZE[1] / 2 - y_offset))
            screen.blit(main_car_image_update, new_pos)
            pygame.draw.circle(screen, 0, new_pos, 5, 0)

            #TODO: Update other car graphics

            # Debug
            draw_box_coords(simulator.environment.main_car, screen, SCREEN_COORD)

            pygame.display.update()
            fpsClock.tick(FPS)


        if t == TIMESTEPS - 1:
            states.append(state)
        time.sleep(SLEEP_DELAY)
