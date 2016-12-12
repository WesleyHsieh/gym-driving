from car import Car
import numpy as np
import pygame

class Environment:
    """
    Coordinates updates to participants
    in environment. Interactions should
    be done through simulator wrapper class.
    """

    def __init__(self, graphics_mode, screen_size, screen=None, terrain=[], num_cpu_cars=4):
        #TODO: Randomize car locations
        cpu_car_textures = ['red', 'orange']
        lims = [[-400.0, 400.0], [-37.5, 37.5]]
        self.main_car = Car(x=0, y=0, angle=0.0, max_vel=20.0, \
            screen=screen, screen_size=screen_size, texture='main')
        self.vehicles = [Car(x=np.random.uniform(lims[0][0], lims[0][1]), y=np.random.uniform(lims[1][0], lims[1][1]), \
            angle=0.0, vel=10.0, screen=screen, screen_size=screen_size, texture=np.random.choice(cpu_car_textures)) \
            for i in range(num_cpu_cars)]
        self.screen_size = screen_size
        self.screen = screen
        self.terrain = terrain
        self.graphics_mode = graphics_mode

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

        #print 'terrain pos', self.terrain[0].get_corners()

    def get_state(self):
        """
        Returns current state, corresponding
        to locations of all cars.
        :return: dict
            'main_car': Position of main car.
            'other_cars': Position of other cars.
        """
        state_dict = {}
        state_dict['main_car'] = self.main_car.get_state()
        state_dict['other_cars'] = [vehicle.get_state() for vehicle in self.vehicles]
        state_dict['car_collisions'] = [self.main_car.collide_rect(car) for car in self.vehicles]
        state_dict['num_car_collisions'] = sum(state_dict['car_collisions'])
        state_dict['terrain_collisions'] = [self.main_car.collide_rect(terrain) for terrain in self.terrain]
        done = False
        return state_dict, done

    def take_action(self, action):
        """
        Takes input action, updates environment.
        :param action: dict
            Input action.
        :return: array
            Reward.
        """
        # TODO: Take in action vector
        terrain_collisions = [terrain for terrain in self.terrain if self.main_car.collide_rect(terrain)]
        car_collisions = [car for car in self.vehicles if self.main_car.collide_rect(car)]
        print "collision textures", [t.texture for t in terrain_collisions]
        print "car collision textures", [c.texture for c in car_collisions]
        self.main_car.take_action(action, terrain_collisions)
        self.step()
        state_dict, done = self.get_state()
        reward = -state_dict['num_car_collisions']
        return state_dict, reward, done
