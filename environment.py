from car import Car
import numpy as np

class Environment:
    """
    Coordinates updates to participants
    in environment. Interactions should
    be done through simulator wrapper class.

    TODO: PyGame graphics
    """

    def __init__(self, num_cpu_cars=2):
        #TODO: Randomize car locations
        self.main_car = Car(0.0, 0.0, 0.0)
        self.vehicles = [Car(0.0, 0.0, 0.0) for _ in range(num_cpu_cars)]
        self.terrain = []

    def step(self):
        """
        Updates environment by one timestep.
        :return: None
        """
        self.main_car.step()
        for vehicle in self.vehicles:
            vehicle.step()

    def get_state(self):
        """
        Returns current state, corresponding
        to locations of all cars.
        :return: dict
            'main_car': Position of main car.
            'other_cars': Position of other cars.
        """
        #TODO: Collision detection
        state_dict = {}
        state_dict['main_car'] = self.main_car.get_state()
        state_dict['other_cars'] = [vehicle.get_state() for vehicle in self.vehicles]
        return state_dict

    def take_action(self, action):
        """
        Takes input action, updates environment.
        :param action: dict
            Input action.
        :return: array
            Reward.
        """
        #TODO: Reward
        self.main_car.take_action(action)
        self.step()
        reward = None
        return reward
