from car import *
from environment import *
from simulator import *

import time
import pygame, sys
from pygame.locals import *

TIMESTEPS = 100
SLEEP_DELAY = .1
STEER_ACTION = 15.0
ACC_ACTION = 2.0
FPS = 30
SCREEN_SIZE = (400, 400)

CAR_X = 0
CAR_Y = 187.5
CAR_ANGLE = 0

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

        # Build the track
        screen.fill(WHITE)
        pygame.draw.line(screen, BLACK, (0, 100), (400, 100), 4)
        pygame.draw.line(screen, BLACK, (0, 300), (400, 300), 4)

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

        # Update Car Location
        CAR_X = next_state['main_car']['x']
        CAR_Y = next_state['main_car']['y']
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
