import numpy as np

class Car:
    """
    Car object.
    """
    def __init__(self, x, y, angle, car_width=50, car_length=25):
        self.x = x
        self.y = y
        self.vel = 0.0
        self.acc = 0.0
        self.angle = angle
        self.car_width = car_width
        length = car_length
        self.car_length = length

    def step(self, t=1):
        """
        Updates car by one timestep.
        :param t: int
            Timestep.
        :return: None
        """
        dist = self.vel * t + 0.5 * self.acc * (t ** 2)
        dx = dist * np.cos(np.radians(self.angle))
        dy = dist * np.sin(np.radians(self.angle))
        self.x += dx
        self.y += dy
        self.vel += self.acc

    def take_action(self, action_dict):
        """
        Updates car state according to action.
        :param action_dict: dict
            'steer': Change in steering angle.
            'acc': Acceleration.
        :return: None
        """
        self.angle = self.angle + action_dict['steer']
        self.angle %= 360.0
        self.acc = action_dict['acc']

    def get_pos(self):
        """
        Returns x, y coordinates.
        :return: tuple
            x, y coordinates.
        """
        return self.x, self.y

    def get_state(self):
        state_dict = {}
        state_dict['x'] = self.x
        state_dict['y'] = self.y
        state_dict['vel'] = self.vel
        state_dict['angle'] = self.angle
        return state_dict

    def get_xpos(self):
        return self.x

    def get_ypos(self):
        return self.y
