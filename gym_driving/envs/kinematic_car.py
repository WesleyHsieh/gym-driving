import numpy as np
import pygame
import os
import IPython

from gym_driving.envs.rectangle import Rectangle
from gym_driving.envs.car import Car

class KinematicCar(Car):
    """
    Car object.
    """
    def __init__(self, x, y, width=50, length=25, angle=0.0, vel=0.0, acc=0.0, max_vel=20.0, mass=100.0, screen=None, screen_size=0, texture='main', graphics_mode=False):
        super(KinematicCar, self).__init__(x, y, width, length, angle, vel, acc, max_vel, mass, screen, screen_size, texture, graphics_mode)
        self.lf = self.lr = width / 2.0

    def step(self, action):
        """
        Updates car by one timestep.
        :param t: int
            Timestep.
        :return: None
        """
        deltaf, a = action
        deltaf = np.radians(deltaf)
        a = max(min(a, self.max_vel - self.vel), -self.vel)
        beta = np.arctan((self.lr / (self.lf + self.lr)) * np.tan(deltaf))
        dx = self.vel * np.cos(np.radians(self.angle) + beta)
        dy = self.vel * np.sin(np.radians(self.angle) + beta)
        dangle = (self.vel / self.lr) * np.sin(beta)
        dvel = a 

        self.x += dx
        self.y += dy
        self.angle += np.rad2deg(dangle)
        self.vel += dvel
        self.angle %= 360.0
        self.acc = max(min(a, self.max_vel - self.vel), -self.vel)
        self.corners = self.calculate_corners()