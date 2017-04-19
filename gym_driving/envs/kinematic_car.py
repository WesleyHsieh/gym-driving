import numpy as np
import pygame
import os
import IPython

from gym_driving.envs.rectangle import Rectangle
from gym_driving.envs.car import Car

class KinematicCar(Car):
    """
    Car object.
    Kinematic bicycle model:
        http://www.me.berkeley.edu/~frborrel/pdfpub/IV_KinematicMPC_jason.pdf
    """
    def __init__(self, x, y, width=50, length=25, angle=0.0, vel=0.0, acc=0.0, max_vel=20.0, mass=100.0, screen=None, screen_size=0, texture='main', graphics_mode=False):
        super(KinematicCar, self).__init__(x, y, width, length, angle, vel, acc, max_vel, mass, screen, screen_size, texture, graphics_mode)
        self.l_f = self.l_r = width / 2.0

    def step(self, action):
        """
        Updates car by one timestep.
        :param t: int
            Timestep.
        :return: None
        """
        # Unpack actions, convert angles to radians
        delta_f, a = action
        delta_f, rad_angle = np.radians(delta_f), np.radians(self.angle)

        # Clamp acceleration if above maximum velocity
        if a > self.max_vel - self.vel:
            a = self.max_vel - self.vel
        elif self.vel + a < 0:
            a = - self.vel

        # Differential equations
        beta = np.arctan((self.l_r / (self.l_f + self.l_r)) * np.tan(delta_f))
        dx = self.vel * np.cos(rad_angle + beta)
        dy = self.vel * np.sin(rad_angle + beta)
        dangle = (self.vel / self.l_r) * np.sin(beta)
        dvel = a 

        # Update car
        self.x += dx
        self.y += dy
        self.angle += np.rad2deg(dangle)
        self.vel += dvel
        self.angle %= 360.0
        self.acc = max(min(a, self.max_vel - self.vel), -self.vel)
        self.corners = self.calculate_corners()