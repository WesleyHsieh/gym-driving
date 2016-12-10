from car import *
from environment import *
from simulator import *
from terrain import *
from controller import *

import time
import pygame, sys
from pygame.locals import *
import random

TIMESTEPS = 10000
SLEEP_DELAY = .05
ACC_ACTION = 5.0
STEER_ACTION = 15.0
FPS = 30
SCREEN_SIZE = (512, 512)
SCREEN_COORD = (0, 0)

CAR_X = 256
CAR_Y = 256
CAR_ANGLE = 0
TERRAINS = []

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAPHICS_MODE = True
CONTROLLER_MODE = 'xbox'

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
    #TERRAINS.append(Terrain(-2048, -256, 1024, 128, 'road', screen, SCREEN_SIZE))
    
    TERRAINS.append(Terrain(-2048, -256, 8192, 128, 'road', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(-2048, -128, 8192, 128, 'grass', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(-2048, 0, 8192, 128, 'road', screen, SCREEN_SIZE))
    # for i in random.sample(xrange(0, 64), 16):
    #     TERRAINS.append(Terrain(-2048 + i*128, 0, 128, 128, 'ice', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(-2048, 128, 8192, 128, 'grass', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(-2048, 256, 8192, 128, 'dirt', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(-2048, 384, 8192, 128, 'grass', screen, SCREEN_SIZE))


    for i in random.sample(xrange(0, 32), 12):
        TERRAINS.append(Terrain(-2048 + i*128, -128 + (i%2) * 128, 128, 128, 'icegrass', screen, SCREEN_SIZE))
    for i in random.sample(xrange(0, 32), 8):
        TERRAINS.append(Terrain(-2048 + i*128, 128 + (i%2) * 128, 128, 128, 'ice', screen, SCREEN_SIZE))
    for i in random.sample(xrange(0, 32), 12):
        TERRAINS.append(Terrain(-2048 + i*128, 384 + (i%2) * 128, 128, 128, 'icegrass', screen, SCREEN_SIZE))


    if GRAPHICS_MODE:
        for t in TERRAINS:
            t.update_graphics(SCREEN_COORD)

        # Add the car
        car_image = pygame.image.load('images/car.png')
        screen.blit(car_image, (CAR_X, CAR_Y))

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

        if GRAPHICS_MODE:
            # Clear the screen
            screen.fill(WHITE)
            car_image_update = pygame.transform.rotate(car_image, -CAR_ANGLE)

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
            screen.blit(car_image_update, new_pos)
            pygame.draw.circle(screen, 0, new_pos, 5, 0)

            # Debug
            draw_box_coords(simulator.environment.main_car, screen, SCREEN_COORD)

            pygame.display.update()
            fpsClock.tick(FPS)


        if t == TIMESTEPS - 1:
            states.append(state)
        time.sleep(SLEEP_DELAY)
