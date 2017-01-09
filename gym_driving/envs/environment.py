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
        self.cpu_car_textures = ['red', 'orange']
        self.num_cpu_cars = num_cpu_cars
        self.screen_size = screen_size
        self.screen = screen
        self.terrain = terrain
        self.graphics_mode = graphics_mode
        self.steer_action = 15.0
        self.acc_action = 5.0
        self.reset()

    def reset(self):
        lims = [[-400.0, 400.0], [-37.5, 37.5]]
        main_car_angle = np.random.choice(np.arange(-30, 31, 15))
        self.main_car = Car(x=0, y=0, angle=main_car_angle, max_vel=20.0, \
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
                collision = any([new_car.collide_rect(car) for car in self.vehicles])
            self.vehicles.append(new_car)
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

        if self.graphics_mode:
            # Clear screen
            self.screen.fill((255, 255, 255))
            main_car_pos = self.main_car.get_pos()
            screen_coord = (main_car_pos[0] - self.screen_size[0]/2, main_car_pos[1] - self.screen_size[1]/2)

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
        state, info_dict['main_car'] = self.main_car.get_state()
        info_dict['other_cars'] = [vehicle.get_state()[1] for vehicle in self.vehicles]
        info_dict['car_collisions'] = [self.main_car.collide_rect(car) for car in self.vehicles]
        info_dict['num_car_collisions'] = sum(info_dict['car_collisions'])
        info_dict['terrain_collisions'] = [self.main_car.collide_rect(terrain) for terrain in self.terrain]
        return state, info_dict

    def take_action(self, action):
        """
        Takes input action, updates environment.
        :param action: dict
            Input action.
        :return: array
            Reward.
        """
        terrain_collisions = [terrain for terrain in self.terrain if self.main_car.collide_rect(terrain)]
        car_collisions = [car for car in self.vehicles if self.main_car.collide_rect(car)]

        # print("collision textures", [t.texture for t in terrain_collisions])
        # print("car collision textures", [c.texture for c in car_collisions])
        # Convert numerical action vector to steering angle / acceleration
        steer = action[0] - 1
        acc = action[1]
        action_unpacked = np.array([steer * self.steer_action, acc * self.acc_action])
        self.main_car.take_action(action_unpacked, terrain_collisions)
        self.step()
        state, info_dict = self.get_state()
        done = any([t.texture == 'grass' for t in terrain_collisions])
        reward = 0 if done else 1

        return state, reward, done, info_dict
