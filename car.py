import numpy as np
from rectangle import Rectangle

class Car(Rectangle):
    """
    Car object.
    """
    def __init__(self, x, y, width=50, length=25, angle=0.0, max_vel=10.0):
        super(Car, self).__init__(x, y, width, length, angle)
        self.vel = 0.0
        self.acc = 0.0
        self.max_vel = max_vel

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
        self.vel = max(min(self.vel, self.max_vel), 0.0)

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
        self.acc = max(min(self.acc, self.max_vel - self.vel), -self.vel)

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
