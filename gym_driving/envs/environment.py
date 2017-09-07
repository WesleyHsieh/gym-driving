from gym_driving.assets.car import Car
from gym_driving.assets.kinematic_car import KinematicCar
from gym_driving.assets.dynamic_car import DynamicCar

from gym_driving.assets.terrain import *

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
        """
        Initializes driving environment interface,
        passes most arguments down to underlying environment.

        Args:
            render_mode: boolean, whether to render.
            screen_size: 1x2 array,
            screen: PyGame screen object, used for rendering.
                Creates own screen object if existing one is not passed in.
            param_dict: dict, param dictionary containing configuration settings.
        """
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
        """
        Resets the environment.

        Returns:
            state: array, state of environment.
        """
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
        # Set main car mass
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
                screen=self.screen, screen_size=self.screen_size, render_mode=self.render_mode).create())
            self.terrain.append(Terrain(x=0, y=0, width=20000, length=200, texture='road', \
                screen=self.screen, screen_size=self.screen_size, render_mode=self.render_mode).create())
            self.terrain.append(Terrain(x=0, y=2000, width=20000, length=3800, texture='grass', \
                screen=self.screen, screen_size=self.screen_size, render_mode=self.render_mode).create())
        else:
            self.terrain = []
            for elem in self.param_dict['terrain_params']:
                if len(elem) == 5:
                    self.terrain.append(Terrain(x=elem[0], y=elem[1], width=elem[2], \
                        length=elem[3], texture=elem[4], screen=self.screen, screen_size=self.screen_size, \
                        render_mode=self.render_mode).create())
                elif len(elem) == 6:
                    self.terrain.append(Terrain(x=elem[0], y=elem[1], width=elem[2], \
                        length=elem[3], angle=elem[4], texture=elem[5], screen=self.screen, \
                        screen_size=self.screen_size, render_mode=self.render_mode).create())
            self.terrain = sorted(self.terrain, key=lambda x: x.friction)
        if self.render_mode:
            self.render()
        self.update_state()
        state, info_dict = self.get_state()
        return state

    def render(self):
        """
        Renders the environment.
        Should only be called if render_mode=True.
        """
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
        """
        Downsamples the input image until it reaches
        the output size using OpenCV's implementation.
        output_size must be smaller or equal to the size
        of the input state by a factor of a multiple of 2.

        Args:
            state: array, input image.
            output_size: 1x2 array, width and height of desired output.

        Returns:
            state: array, output downsampled image.
        """
        w, h = state.shape
        if output_size is not None:
            while w > output_size and h > output_size:
                state = cv2.pyrDown(state, dstsize = (w / 2, h / 2))
                w, h = state.shape
        return state

    def get_state(self):
        """
        Returns current stored state and info dict.

        Returns:
            state: array, state of environment.
                Can be positions and angles of cars, or image of environment
                depending on configuration.
            info_dict: dict, contains information about environment that may
                not be included in the state.
        """
        return self.state, self.info_dict

    def update_state(self):
        """
        Updates current stored state and info dict.
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
        """
        Returns current internal state of the cars in the environment.
        Output state is used to set internal state of environment.

        Returns:
            main_car_info_dict: dict, contains info dict with internal state of main car.
            vehicle_info_dicts: list, contains list of info dicts with internal states of CPU cars.
        """
        _, main_car_info_dict = self.main_car.get_state()
        if self.num_cpu_cars > 0:
            vehicle_info_dicts = list(zip(*[car.get_state() for car in self.vehicles]))[1]
        else:
            vehicle_info_dicts = []
        return main_car_info_dict, vehicle_info_dicts

    def set_state(self, main_car_state, vehicles_states):
        """
        Sets state of all cars in the environment based
        on input states, obtained by get_compact_state().

        Args:
            main_car_state: dict, contains info dict with internal state of main car.
            vehicles_states: list, contains list of info dicts with internal states of CPU cars.
        """
        self.main_car.set_state(**main_car_state)
        for i in range(len(self.vehicles)):
            self.vehicles[i].set_state(**vehicles_states[i])


    def step(self, action, render_mode=None):
        """
        Updates the environment for one step.

        Args:
            action: 1x2 array, steering / acceleration action.

        Returns:
            state: array, state of environment.
                Can be positions and angles of cars, or image of environment
                depending on configuration.
            reward: float, reward from action taken.
            done: boolean, whether trajectory is finished.
            info_dict: dict, contains information about environment that may
                not be included in the state.
        """
        # Create Noise
        #noise=param_dict['noise'] ## (type, magnitude)
        # Convert numerical action vector to steering angle / acceleration
        if self.control_space == 'discrete':
            if type(action) is int:
                steer = self.steer_space[action]
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
        noise = self.param_dict['noise']
        if noise[0] == 'gaussian': #applies gaussian noise whenever an action is taken
            if steer != 0.0 and noise > 0.0:
                steer *= 1.0 + np.random.normal(loc=0.0, scale=noise[1])
            if acc != 0.0 and noise > 0.0:
                acc *= 1.0 + np.random.normal(loc=0.0, scale=noise[1])
        elif noise[0] == 'random': #uniformly samples from the action space
            if np.random.uniform() >= 0.1:
                steer = np.random.uniform(-1, 1) * self.param_dict['steer_action'][1]
                acc = np.random.uniform(-1, 1) * self.param_dict['acc_action'][1]

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
        """
        Simulate a sequence of actions.

        Args:
            noise: float, standard deviation of zero-mean Gaussian noise
            state: dict, internal starting state of environment.
                Currently set as the positions, velocities, and angles of
                all cars.

        Returns:
            states: list, list of states in trajectory.
            rewards: list, list of rewards in trajectory.
            dones: list, list of dones in trajectory.
            info_dicts: list, list of info dicts in trajectory.
        """
        # Save copy of original states
        _, main_car_info_dict = self.main_car.get_state()
        vehicle_info_dicts = list(zip(*[car.get_state() for car in self.vehicles]))[1]

        if compact_state is not None:
            main_car_state, vehicles_states = compact_state
            self.set_state(main_car_state, vehicles_states)

        # Take actions
        states, rewards, dones, info_dicts, = [], [], [], []
        for action in actions:
            state, reward, done, info_dict = self.step([action, 1], noise=noise, render_mode=False)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            info_dicts.append(info_dict)

        # Restore cars to original states
        self.set_state(main_car_info_dict, vehicle_info_dicts)

        # Return new state after taking actions
        return states, rewards, dones, info_dicts
