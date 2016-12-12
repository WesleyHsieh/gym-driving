from car import Car
import numpy as np

class Environment:
    """
    Coordinates updates to participants
    in environment. Interactions should
    be done through simulator wrapper class.
    """

    def __init__(self, screen_size, terrain=[], num_cpu_cars=2):
        #TODO: Randomize car locations
        self.main_car = Car(0, 0)
        self.vehicles = [Car(0, -100 + 200 * i) for i in range(num_cpu_cars)]
        self.terrain = terrain

    def step(self):
        """
        Updates environment by one timestep.
        :return: None
        """
        self.main_car.step()
        for vehicle in self.vehicles:
            vehicle.step()
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
        terrain_collisions = [terrain for terrain in self.terrain if self.main_car.collide_rect(terrain)]
        self.main_car.take_action(action, terrain_collisions)
        self.step()
        state_dict, done = self.get_state()
        reward = -state_dict['num_car_collisions']
        return state_dict, reward, done
