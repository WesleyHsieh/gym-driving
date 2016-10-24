from car import *
from environment import *
from simulator import *

import time
import pygame
from pygame.locals import *

TIMESTEPS = 20
SLEEP_DELAY = 1
STEER_ACTION = 15.0
ACC_ACTION = 2.0

def process_keys():
    action_dict = {'steer': 0.0, 'acc': 0.0}
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        action_dict['acc'] = ACC_ACTION
    elif keys[pygame.K_DOWN]:
        action_dict['acc'] = -ACC_ACTION
    if keys[pygame.K_LEFT]:
        action_dict['steer'] = STEER_ACTION
    elif keys[pygame.K_RIGHT]:
        action_dict['steer'] = -STEER_ACTION
    return action_dict

if __name__ == '__main__':
    screen_size = (400, 400)
    screen = pygame.display.set_mode(screen_size)
    screen.fill((0,192,0))

    simulator = Simulator()
    states, actions, rewards = [], [], []
    for t in range(TIMESTEPS):
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

        if t == TIMESTEPS - 1:
            states.append(next_state)
        time.sleep(SLEEP_DELAY)

    #print "States"
    #print states
    #print "Actions"
    #print actions
    #print "Rewards"
    #print rewards