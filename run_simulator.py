from car import *
from environment import *
from simulator import *
from terrain import *

import time
import pygame, sys
from pygame.locals import *
import random

TIMESTEPS = 10000
SLEEP_DELAY = .05
STEER_ACTION = 15.0
ACC_ACTION = 5.0
FPS = 30
SCREEN_SIZE = (512, 512)
SCREEN_COORD = (0, 0)

CAR_X = 256
CAR_Y = 256
CAR_ANGLE = 0
TERRAINS = []

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def process_keys():
    action_dict = {'steer': 0.0, 'acc': 0.0}
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        action_dict['acc'] = ACC_ACTION
    elif keys[pygame.K_DOWN]:
        action_dict['acc'] = -ACC_ACTION
    if keys[pygame.K_LEFT]:
        action_dict['steer'] = -STEER_ACTION
    elif keys[pygame.K_RIGHT]:
        action_dict['steer'] = +STEER_ACTION
    return action_dict

def draw_box_coords(rectangle, screen, SCREEN_COORD):
    corners = rectangle.get_corners()
    for c in corners:
        pos = (int(c[0] - SCREEN_COORD[0]), int(c[1] - SCREEN_COORD[1]))
        pygame.draw.circle(screen, 0, pos, 5, 0)
    c = rectangle.get_pos()
    pos = (int(c[0] - SCREEN_COORD[0]), int(c[1] - SCREEN_COORD[1]))
    pygame.draw.circle(screen, 0, pos, 5, 0)

if __name__ == '__main__':
    # PyGame initilizations
    pygame.init()
    fpsClock = pygame.time.Clock()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('Driving Simulator')

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

    '''
    for i in random.sample(xrange(0, 32), 12):
        TERRAINS.append(Terrain(-2048 + i*128, -128 + (i%2) * 128, 128, 128, 'icegrass', screen, SCREEN_SIZE))
    for i in random.sample(xrange(0, 32), 8):
        TERRAINS.append(Terrain(-2048 + i*128, 128 + (i%2) * 128, 128, 128, 'ice', screen, SCREEN_SIZE))
    for i in random.sample(xrange(0, 32), 12):
        TERRAINS.append(Terrain(-2048 + i*128, 384 + (i%2) * 128, 128, 128, 'icegrass', screen, SCREEN_SIZE))
    '''

    for t in TERRAINS:
        t.update_graphics(SCREEN_COORD)

    # Add the car
    car_image = pygame.image.load('images/car.png')
    screen.blit(car_image, (CAR_X, CAR_Y))

    simulator = DrivingEnv(SCREEN_SIZE, TERRAINS)
    states, actions, rewards = [], [], []

    for t in range(TIMESTEPS):
        # Checks for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Clear the screen
        screen.fill(WHITE)
        key = pygame.key.get_pressed()
        action = process_keys()

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
        car_image_update = pygame.transform.rotate(car_image, -CAR_ANGLE)

        # Update SCREEN_COORD
        SCREEN_COORD = (CAR_X - SCREEN_SIZE[0]/2, CAR_Y - SCREEN_SIZE[1]/2)

        # Update terrain graphics
        for t in TERRAINS:
            t.update_graphics(SCREEN_COORD)
            draw_box_coords(t, screen, SCREEN_COORD)


        # Update car graphics
        # TODO: Blit currently maps to top left corner
        car = simulator.environment.main_car
        corners = car.get_corners()
        center = car.get_pos()
        top_left = corners[0]
        x_offset, y_offset = top_left[0] - center[0], top_left[1] - center[1]
        new_pos = (int(SCREEN_SIZE[0] / 2 - x_offset), int(SCREEN_SIZE[1] / 2 - y_offset))
        screen.blit(car_image_update, new_pos)

        # Debug
        # circle(Surface, color, pos, radius, width=0)
        draw_box_coords(simulator.environment.main_car, screen, SCREEN_COORD)

        pygame.display.update()
        fpsClock.tick(FPS)


        if t == TIMESTEPS - 1:
            states.append(state)
        time.sleep(SLEEP_DELAY)
