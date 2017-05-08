import numpy as np
import pygame
import os
import IPython
from scipy.integrate import odeint

from gym_driving.envs.rectangle import Rectangle
from gym_driving.envs.car import Car

class KinematicCar(Car):
    """
    Car object.
    Kinematic bicycle model:
        http://www.me.berkeley.edu/~frborrel/pdfpub/IV_KinematicMPC_jason.pdf
    """
    def __init__(self, x, y, width=50, length=25, angle=0.0, vel=0.0, acc=0.0, max_vel=20.0, mass=100.0, screen=None, screen_size=0, texture='main', render_mode=False):
        super(KinematicCar, self).__init__(x, y, width, length, angle, vel, acc, max_vel, mass, screen, screen_size, texture, render_mode)
        self.l_f = self.l_r = width / 2.0

    def step(self, action, info_dict=None):
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
        ode_state = [self.x, self.y, self.vel, rad_angle]
        aux_state = (a, delta_f)
        t = np.arange(0.0, 1.0, 0.1)
        delta_ode_state = odeint(self.integrator, ode_state, t, args=aux_state)
        x, y, vel, angle = delta_ode_state[-1]

        # Update car
        self.x, self.y, self.vel, self.angle = x, y, vel, np.rad2deg(angle)
        self.angle %= 360.0
        self.corners = self.calculate_corners()

    def integrator(self, state, t, acc, delta_f):
        x, y, vel, rad_angle = state

        # Differential equations
        beta = np.arctan((self.l_r / (self.l_f + self.l_r)) * np.tan(delta_f))
        dx = vel * np.cos(rad_angle + beta)
        dy = vel * np.sin(rad_angle + beta)
        dangle = (vel / self.l_r) * np.sin(beta)
        dvel = acc
        output = [dx, dy, dvel, dangle]
        return output

