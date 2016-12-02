import numpy as np
from rectangle import Rectangle

class Car(Rectangle):
    """
    Car object.
    """
    def __init__(self, x, y, width=50, length=25, angle=0.0, max_vel=20.0, mass=100.0):
        super(Car, self).__init__(x, y, width, length, angle)
        self.vel_angle = angle
        self.vel = 0.0
        self.acc = 0.0
        self.max_vel = max_vel
        self.mass = mass

    def step(self, t=1):
        """
        Updates car by one timestep.
        :param t: int
            Timestep.
        :return: None
        """
        dist = self.vel * t + 0.5 * self.acc * (t ** 2)
        dx = dist * np.cos(np.radians(self.vel_angle))
        dy = dist * np.sin(np.radians(self.vel_angle))
        self.x += dx
        self.y += dy
        self.vel += self.acc
        self.vel = max(min(self.vel, self.max_vel), 0.0)

    def take_action(self, action_dict, terrain_collisions):
        """
        Updates car state according to action.
        :param action_dict: dict
            'steer': Change in steering angle.
            'acc': Acceleration.
        :return: None
        """
        # Get properties of terrain that the car is currently on
        decel = np.sum([t.decel for t in terrain_collisions])
        slip = np.sum([t.slip for t in terrain_collisions])
        print "decel", decel
        print "slip", slip

        if slip == 0:
            self.vel_angle = self.angle
        self.angle += action_dict['steer']
        self.angle %= 360.0
        self.acc = action_dict['acc'] - decel
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
        state_dict['vel_angle'] = self.vel_angle
        return state_dict

    def get_xpos(self):
        return self.x

    def get_ypos(self):
        return self.y
