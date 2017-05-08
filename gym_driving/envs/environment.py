from gym_driving.envs.car import Car
from gym_driving.envs.kinematic_car import KinematicCar
from gym_driving.envs.dynamic_car import DynamicCar

from gym_driving.envs.terrain import *

import numpy as np
import pygame
import cv2 
import IPython

class Environment:
    """
    Coordinates updates to participants
    in environment. Interactions should
    be done through simulator wrapper class.
    """

    def __init__(self, render_mode, screen_size, screen=None, param_dict=None):
        self.param_dict = param_dict
        self.cpu_car_textures = ['orange', 'red']
        self.screen_size = screen_size
        self.screen = screen
        self.render_mode = render_mode
        self.steer_action = self.param_dict['steer_action']
        self.acc_action = self.param_dict['acc_action']
        self.downsampled_size = self.param_dict['downsampled_size']
        self.state_space = self.param_dict['state_space']
        self.control_space = self.param_dict['control_space']
        self.main_car_types = {
            'point': Car,
            'kinematic': KinematicCar,
            'dynamic': DynamicCar,
        }
        self.reset()

    def reset(self, screen=None):
        if screen is not None:
            self.screen = screen
        self.num_cpu_cars = self.param_dict['num_cpu_cars']
        lims = self.param_dict['cpu_cars_bounding_box']

        # Main car starting angle
        low, high, num = self.param_dict['main_car_starting_angles']
        if num is None:
            main_car_angle = np.random.uniform(low=low, high=high)
        else:
            main_car_angle = np.random.choice(np.linspace(low, high, num))

        # Control space
        if self.param_dict['control_space'] == 'discrete':
            low, high, num = self.param_dict['steer_action']
            self.steer_space = np.linspace(low, high, num)
            low, high, num = self.param_dict['acc_action']
            self.acc_space = np.linspace(low, high, num)
        else:
            low, high, num = self.param_dict['steer_action']
            self.steer_space = [low, high]
            low, high, num = self.param_dict['acc_action']
            self.acc_space = [low, high]

        x, y, vel, max_vel = self.param_dict['main_car_params']
        main_car_dynamics = self.main_car_types[self.param_dict['main_car_dynamics']]
        self.main_car = main_car_dynamics(x=x, y=y, angle=main_car_angle, vel=vel, max_vel=max_vel, \
            screen=self.screen, screen_size=self.screen_size, texture='main', \
            render_mode=self.render_mode)

        # Create CPU-controlled cars, ensuring they are collision-free
        self.vehicles = []
        for _ in range(self.num_cpu_cars):
            collision = True
            while collision:
                new_car = Car(x=np.random.uniform(lims[0][0], lims[0][1]), y=np.random.uniform(lims[1][0], lims[1][1]), \
                angle=0.0, vel=10.0, screen=self.screen, screen_size=self.screen_size, \
                texture=np.random.choice(self.cpu_car_textures), render_mode=self.render_mode)
                collision = any([new_car.collide_rect(car) for car in self.vehicles]) or new_car.collide_rect(self.main_car)
            self.vehicles.append(new_car)
        
        if self.param_dict['terrain_params'] is None:
            self.terrain = []
            self.terrain.append(Terrain(x=0, y=-2000, width=20000, length=3800, texture='grass', \
                screen=self.screen, screen_size=self.screen_size, render_mode=self.render_mode))
            self.terrain.append(Terrain(x=0, y=0, width=20000, length=200, texture='road', \
                screen=self.screen, screen_size=self.screen_size, render_mode=self.render_mode))
            self.terrain.append(Terrain(x=0, y=2000, width=20000, length=3800, texture='grass', \
                screen=self.screen, screen_size=self.screen_size, render_mode=self.render_mode))
        else:
            self.terrain = [Terrain(x=x, y=y, width=width, length=length, texture=texture, \
                screen=self.screen, screen_size=self.screen_size, render_mode=self.render_mode) \
                for x, y, width, length, texture in self.param_dict['terrain_params']]
            self.terrain = sorted(self.terrain, key=lambda x: x.friction)
        if self.render_mode:
            self.render()
        self.update_state()
        state, info_dict = self.get_state()
        return state

    def render(self):
        # Clear screen
        self.screen.fill((255, 255, 255))
        main_car_pos = self.main_car.get_pos()
        screen_coord = (main_car_pos[0] - self.screen_size[0]/2, main_car_pos[1] - self.screen_size[1]/2)

        # Update terrain, vehicles
        for t in self.terrain:
            t.render(screen_coord)
        self.main_car.render(screen_coord)
        for c in self.vehicles:
            c.render(screen_coord)
        
        pygame.display.update()

    def downsample(self, state, output_size):
        w, h = state.shape
        if output_size is not None:
            while w > output_size and h > output_size:
                state = cv2.pyrDown(state, dstsize = (w / 2, h / 2))
                w, h = state.shape
        return state

    def get_state(self):
        return self.state, self.info_dict

    def update_state(self):
        """
        Returns current state, corresponding
        to locations of all cars.
        :return: dict
            'main_car': Position of main car.
            'other_cars': Position of other cars.
        """
        info_dict = {}
        main_car_state, info_dict['main_car'] = self.main_car.get_state()
        # x = [vehicle.get_state() for vehicle in self.vehicles]
        if self.num_cpu_cars > 0:
            car_states, info_dict['other_cars'] = list(zip(*[vehicle.get_state() for vehicle in self.vehicles]))
        info_dict['car_collisions'] = [car for car in self.vehicles if self.main_car.collide_rect(car)]
        # info_dict['num_car_collisions'] = len(info_dict['car_collisions'])
        info_dict['terrain_collisions'] = [terrain for terrain in self.terrain if self.main_car.collide_rect(terrain)]
        # info_dict['angle_diff'] = abs(main_car_state[2])
        # info_dict['distance_to_nearest_car'] = min([self.main_car.distance_to_rectangle(vehicle) for vehicle in self.vehicles])

        # Compact state
        info_dict['compact_state'] = self.get_compact_state()

        if self.state_space == 'positions':
            if self.num_cpu_cars > 0:
                # state = np.concatenate([main_car_state] + list(car_states))
                filtered_car_states = np.concatenate([car_state[:2] - main_car_state[:2] \
                    for car_state in car_states], axis=0)
                screen_lims = self.screen_size[0]
                filtered_car_states[filtered_car_states > screen_lims] = screen_lims
                filtered_car_states[filtered_car_states < -screen_lims] = -screen_lims
                main_car_state[0] = 0.0
                state = np.concatenate([main_car_state, filtered_car_states])
            else:
                state = main_car_state
        elif self.state_space == 'image':
            state = pygame.surfarray.array2d(self.screen).astype(np.uint8)
            self.downsample(state, self.downsampled_size)
        self.state, self.info_dict = state, info_dict

    def get_compact_state(self):
        _, main_car_info_dict = self.main_car.get_state() 
        if self.num_cpu_cars > 0:
            vehicle_info_dicts = list(zip(*[car.get_state() for car in self.vehicles]))[1]
        else:
            vehicle_info_dicts = []
        return (main_car_info_dict, vehicle_info_dicts)

    def set_state(self, main_car_state, vehicles_states):
        self.main_car.set_state(**main_car_state)
        for i in range(len(self.vehicles)):
            self.vehicles[i].set_state(**vehicles_states[i])

    def step(self, action, noise=0.1, render_mode=None):
        """
        Takes input action, updates environment.
        :param action: dict
            Input action.
        :return: array
            Reward.
        """
        # Convert numerical action vector to steering angle / acceleration
        if self.control_space == 'discrete':
            if type(action) is int:
                steer = action 
                acc = 0.0
            else:
                steer = self.steer_space[action[0]]
                acc = self.acc_space[action[1]]
        elif self.control_space == 'continuous':
            if type(action) is float:
                steer = action
                acc = 0.0
            else:
                steer = action[0]
                acc = action[1]

        # Add noise
        if steer != 0.0 and noise > 0.0:
            steer *= 1.0 + np.random.normal(loc=0.0, scale=noise)
        if acc != 0.0 and noise > 0.0:
            acc *= 1.0 + np.random.normal(loc=0.0, scale=noise)

        # Convert to action space, apply action
        action_unpacked = np.array([steer, acc])
        
        # Get old state, step
        state, info_dict = self.get_state()
        self.main_car.step(action_unpacked, info_dict)
        for vehicle in self.vehicles:
            vehicle.step(None)

        self.update_state()
        state, info_dict = self.get_state()
        terrain_collisions = info_dict['terrain_collisions']
        car_collisions = info_dict['car_collisions']
        done = any([t.texture == 'grass' for t in terrain_collisions]) or len(car_collisions) >= 1
        reward = 0 if done else 1

        if render_mode or (render_mode is None and self.render_mode):
            self.render()

        return state, reward, done, info_dict

    def simulate_actions(self, actions, noise=0.0, compact_state=None):
        # Save copy of original states
        _, main_car_info_dict = self.main_car.get_state() 
        vehicle_info_dicts = list(zip(*[car.get_state() for car in self.vehicles]))[1]

        if compact_state is not None:
            main_car_state, vehicles_states = compact_state
            self.set_state(main_car_state, vehicles_states)

        # Take actions
        states, rewards, dones, info_dicts, = [], [], [], []
        for action in actions:
            state, reward, done, info_dict = self.step([action, 1.0], noise=noise, render_mode=False)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            info_dicts.append(info_dict)
        
        # Restore cars to original states
        self.set_state(main_car_info_dict, vehicle_info_dicts)

        # Return new state after taking actions 
        return states, rewards, dones, info_dicts

