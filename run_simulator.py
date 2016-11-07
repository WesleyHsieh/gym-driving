from car import *
from environment import *
from simulator import *

import time
import pygame, sys
from pygame.locals import *

TIMESTEPS = 200
SLEEP_DELAY = .1
STEER_ACTION = 15.0
ACC_ACTION = 2.0
FPS = 30
SCREEN_SIZE = (512, 512)
SCREEN_COORD = (0, 0)

CAR_X = 256
CAR_Y = 256
CAR_ANGLE = 0

GRASS_COORD = []
ROAD_COORD = []

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

if __name__ == '__main__':
    # PyGame initilizations
    pygame.init()
    fpsClock = pygame.time.Clock()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('Driving Simulator')

    # Add all terrain coordinates
    for i in range(0, 8):
        GRASS_COORD.append((i*128, 0))
        ROAD_COORD.append((i*128, 128))
        ROAD_COORD.append((i*128, 256))
        GRASS_COORD.append((i*128, 384))

    # Add the terrain
    grass_tile = pygame.image.load('images/grass_tile_lite.jpg')
    road_tile = pygame.image.load('images/road_tile_lite.jpg')
    for coord in GRASS_COORD:
        if coord[0] - SCREEN_COORD[0] < 512 and coord[1] - SCREEN_COORD[1] < 512:
            screen.blit(grass_tile, coord)
    for coord in ROAD_COORD:
        if coord[0] - SCREEN_COORD[0] < 512 and coord[1] - SCREEN_COORD[1] < 512:
            screen.blit(road_tile, coord)

    # Add the car
    car_image = pygame.image.load('images/car.png')
    screen.blit(car_image, (CAR_X, CAR_Y))

    simulator = Simulator()
    states, actions, rewards = [], [], []
    for t in range(TIMESTEPS):
        # Checks for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Clear the screen
        screen.fill(WHITE)

        state = simulator.get_state()

        key = pygame.key.get_pressed()
        action = process_keys()

        reward = simulator.take_action(action)
        next_state = simulator.get_state()

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        print "State, Action, Next State"
        print state
        print action
        print next_state

        # Update SCREEN_COORD
        SCREEN_COORD = (next_state['main_car']['x'] - 256, next_state['main_car']['y'] - 256)

        # Draw tiles that screen is in
        for coord in GRASS_COORD:
            if coord[0] - SCREEN_COORD[0] < 512 and coord[1] - SCREEN_COORD[1] < 512:
                screen.blit(grass_tile, (coord[0] - SCREEN_COORD[0], coord[1] - SCREEN_COORD[1]))
        for coord in ROAD_COORD:
            if coord[0] - SCREEN_COORD[0] < 512 and coord[1] - SCREEN_COORD[1] < 512:
                screen.blit(road_tile, (coord[0] - SCREEN_COORD[0], coord[1] - SCREEN_COORD[1]))

        # Update Car Location
        #CAR_X = next_state['main_car']['x']
        #CAR_Y = next_state['main_car']['y']
        CAR_ANGLE = next_state['main_car']['angle']
        car_image_update = pygame.transform.rotate(car_image, -CAR_ANGLE)
        screen.blit(car_image_update, (CAR_X, CAR_Y))

        pygame.display.update()
        fpsClock.tick(FPS)

        if t == TIMESTEPS - 1:
            states.append(next_state)
        time.sleep(SLEEP_DELAY)

    #print "States"
    #print states
    #print "Actions"
    #print actions
    #print "Rewards"
    #print rewards
