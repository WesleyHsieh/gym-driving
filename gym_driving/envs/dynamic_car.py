import numpy as np
import pygame
import os

from gym_driving.envs.rectangle import Rectangle
from gym_driving.envs.car import Car

class DynamicCar(Car):
    """
    Car object.
    """
    def __init__(self, x, y, width=50, length=25, angle=0.0, vel=0.0, acc=0.0, max_vel=20.0, mass=100.0, screen=None, screen_size=0, texture='main', graphics_mode=False):
        super(DynamicCar, self).__init__(x, y, width, length, angle, vel, acc, max_vel, mass, screen, screen_size, texture, graphics_mode)
        self.mass = 1000.0
    # def step(self, t=1):
    #     """
    #     Updates car by one timestep.
    #     :param t: int
    #         Timestep.
    #     :return: None
    #     """
    #     dist = self.vel * t + 0.5 * self.acc * (t ** 2)
    #     dx = dist * np.cos(np.radians(self.vel_angle))
    #     dy = dist * np.sin(np.radians(self.vel_angle))
    #     self.x += dx
    #     self.y += dy
    #     self.vel += self.acc
    #     self.vel = max(min(self.vel, self.max_vel), 0.0)
    #     self.corners = self.calculate_corners()

    def step(self, action):
        # Inputs: alpha (slip angle)
        fz = 9.81 * self.mass
        tw = 7.3
        tp = 24
        fzt = 980
        c1, c2, c3, c4 = 1.0, 0.34, 0.57, 0.32
        a0, a1, a2, a3, a4 = 1068, 11.3, 2442.73, 0.31, -1877
        ka, k1 = 0.05, 0.000008
        cs_fz = 17.91
        # Friction, depends on road
        mu_0 = mu = 0.85

        apo = 0.0768 * np.sqrt(fz * fzt) / (tw * (tp + 5))
        ap = apo * (1 - ka * fx / fz)
        ks = 2 / (apo ** 2) * (a0 + a1 * fz - (a1 / a2) * (fz ** 2))
        kc = 2 / (apo ** 2) * fz * cs_fz
        sigma = np.pi * (ap ** 2) / (8 * mu_0 * fz) * \
            np.sqrt((ks ** 2) * (np.tan(alpha) ** 2) + (kc ** 2) * (s / (1 - s)) ** 2)
        f_sigma = (c1 * sigma ** 3 + c2 * sigma ** 2 + 4 / np.pi * sigma) / \
            (c1 * sigma ** 3 + c3 * sigma ** 2 + c4 * sigma + 1)
        fc = f_sigma * mu * fz
    # def get_state(self):
    #     return super(KinematicCar, self).get_state()

    # def set_state(self, x, y, vel, angle, vel_angle):
    #     self.x = x
    #     self.y = y
    #     self.vel = vel
    #     self.angle = angle
    #     self.vel_angle = vel_angle
