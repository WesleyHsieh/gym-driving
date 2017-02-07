from gym_driving.envs.car import Car

import numpy as np
import pygame

class Environment:
    """
    Coordinates updates to participants
    in environment. Interactions should
    be done through simulator wrapper class.
    """

    def __init__(self, graphics_mode, screen_size, screen=None, terrain=[], num_cpu_cars=5):
        self.cpu_car_textures = ['blue', 'green']
        self.num_cpu_cars = num_cpu_cars
        self.screen_size = screen_size
        self.screen = screen
        self.terrain = terrain
        self.graphics_mode = graphics_mode
        self.steer_action = 15.0
        self.acc_action = 5.0
        self.reset()

    def reset(self):
        lims = [[100.0, 1000.0], [-90.0, 90.0]]
        main_car_angle = np.random.choice(np.arange(-30, 31, 15))
        self.main_car = Car(x=0.0, y=0.0, angle=main_car_angle, max_vel=20.0, \
            screen=self.screen, screen_size=self.screen_size, texture='main', \
            graphics_mode=self.graphics_mode)
        # Create CPU-controlled cars, ensuring they are collision-free
        self.vehicles = []
        for _ in range(self.num_cpu_cars):
            collision = True
            while collision:
                new_car = Car(x=np.random.uniform(lims[0][0], lims[0][1]), y=np.random.uniform(lims[1][0], lims[1][1]), \
                angle=0.0, vel=10.0, screen=self.screen, screen_size=self.screen_size, \
                texture=np.random.choice(self.cpu_car_textures), graphics_mode=self.graphics_mode)
                collision = any([new_car.collide_rect(car) for car in self.vehicles]) or new_car.collide_rect(self.main_car)
            self.vehicles.append(new_car)
        if self.graphics_mode:
            self.update_graphics()
        state, info_dict = self.get_state()
        return state

    def step(self):
        """
        Updates environment by one timestep.
        :return: None
        """
        self.main_car.step()
        for vehicle in self.vehicles:
            vehicle.step()

    def update_graphics(self):
        # Clear screen
        self.screen.fill((255, 255, 255))
        main_car_pos = self.main_car.get_pos()
        screen_coord = (main_car_pos[0] - self.screen_size[0]/2, main_car_pos[1] - self.screen_size[1]/2)

        # Update terrain, vehicles
        for t in self.terrain:
            t.update_graphics(screen_coord)
        self.main_car.update_graphics(screen_coord)
        for c in self.vehicles:
            c.update_graphics(screen_coord)
        
        pygame.display.update()

    def get_state(self):
        """
        Returns current state, corresponding
        to locations of all cars.
        :return: dict
            'main_car': Position of main car.
            'other_cars': Position of other cars.
        """
        info_dict = {}
        main_car_state, info_dict['main_car'] = self.main_car.get_state()
        x = [vehicle.get_state() for vehicle in self.vehicles]
        car_states, info_dict['other_cars'] = list(zip(*[vehicle.get_state() for vehicle in self.vehicles]))
        info_dict['car_collisions'] = [car for car in self.vehicles if self.main_car.collide_rect(car)]
        # info_dict['num_car_collisions'] = len(info_dict['car_collisions'])
        info_dict['terrain_collisions'] = [terrain for terrain in self.terrain if self.main_car.collide_rect(terrain)]
        # info_dict['angle_diff'] = abs(main_car_state[2])
        # info_dict['distance_to_nearest_car'] = min([self.main_car.distance_to_rectangle(vehicle) for vehicle in self.vehicles])

        # Compact state
        info_dict['compact_state'] = self.get_compact_state()

        state = np.concatenate([main_car_state] + list(car_states))
        return state, info_dict

    def get_compact_state(self):
        _, main_car_info_dict = self.main_car.get_state() 
        vehicle_info_dicts = list(zip(*[car.get_state() for car in self.vehicles]))[1]
        return (main_car_info_dict, vehicle_info_dicts)

    def set_state(self, main_car_state, vehicles_states):
        self.main_car.set_state(**main_car_state)
        for i in range(len(self.vehicles)):
            self.vehicles[i].set_state(**vehicles_states[i])

    def take_action(self, action, noise=0.1, graphics_mode=None):
        """
        Takes input action, updates environment.
        :param action: dict
            Input action.
        :return: array
            Reward.
        """
        # Convert numerical action vector to steering angle / acceleration
        # steer = action[0] - 1
        steer = (action[0] - 2) / 2.0
        acc = action[1]

        # Add noise
        if steer != 0.0 and noise > 0.0:
            steer += np.random.normal(loc=0.0, scale=noise)
        if acc != 0.0 and noise > 0.0:
            acc += np.random.normal(loc=0.0, scale=noise)

        # Convert to action space, apply action
        action_unpacked = np.array([steer * self.steer_action, acc * self.acc_action])
        self.main_car.take_action(action_unpacked)
        self.step()
        state, info_dict = self.get_state()
        terrain_collisions = info_dict['terrain_collisions']
        car_collisions = info_dict['car_collisions']
        done = any([t.texture == 'grass' for t in terrain_collisions]) or len(car_collisions) >= 1
        reward = 0 if done else 1

        if graphics_mode or (graphics_mode is None and self.graphics_mode):
            self.update_graphics()

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
            state, reward, done, info_dict = self.take_action([action, 1.0], noise=noise, graphics_mode=False)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            info_dicts.append(info_dict)
        
        # Restore cars to original states
        self.set_state(main_car_info_dict, vehicle_info_dicts)

        # Return new state after taking actions 
        return states, rewards, dones, info_dicts

