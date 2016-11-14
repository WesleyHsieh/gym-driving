from car import *
from environment import *
from simulator import *
from terrain import *

import time
import pygame, sys
from pygame.locals import *

TIMESTEPS = 200
SLEEP_DELAY = .05
STEER_ACTION = 15.0
ACC_ACTION = 2.0
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

if __name__ == '__main__':
    # PyGame initilizations
    pygame.init()
    fpsClock = pygame.time.Clock()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('Driving Simulator')

    # Add the terrain
    TERRAINS.append(Terrain(-4096, -4096, 8192, 4224, 'grass', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(-4096, 128, 8192, 256, 'road', screen, SCREEN_SIZE))
    TERRAINS.append(Terrain(-4096, 384, 8192, 4224, 'grass', screen, SCREEN_SIZE))
    for t in TERRAINS:
        t.update_graphics(SCREEN_COORD)

    # Add the car
    car_image = pygame.image.load('images/car.png')
    screen.blit(car_image, (CAR_X, CAR_Y))

    simulator = DrivingEnv(SCREEN_SIZE)
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
        print "State, Action, Next State"
        print states[t-1]
        print action
        print states[t]

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

        # Update car graphics
        screen.blit(car_image_update, tuple(x/2 for x in SCREEN_SIZE))

        pygame.display.update()
        fpsClock.tick(FPS)

        if t == TIMESTEPS - 1:
            states.append(state)
        time.sleep(SLEEP_DELAY)

    #print "States"
    #print states
    #print "Actions"
    #print actions
    #print "Rewards"
    #print rewards