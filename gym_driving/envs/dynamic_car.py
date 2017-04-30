import numpy as np
import pygame
import os
import IPython

from gym_driving.envs.rectangle import Rectangle
from gym_driving.envs.car import Car

class DynamicCar(Car):
    """
    Car object.
    Dynamic bicycle model:
        http://www.me.berkeley.edu/~frborrel/pdfpub/IV_KinematicMPC_jason.pdf
    Cornering stiffness calculation:
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.46.28&rep=rep1&type=pdf
    Yaw inertia estimate:
        https://www.degruyter.com/downloadpdf/j/mecdc.2013.11.issue-1/mecdc-2013-0003/mecdc-2013-0003.pdf
    Friction coefficients:
        http://www.gipsa-lab.grenoble-inp.fr/~moustapha.doumiati/MED2010.pdf
    """
    def __init__(self, x, y, width=50, length=25, angle=0.0, vel=0.0, acc=0.0, max_vel=20.0, mass=100.0, screen=None, screen_size=0, texture='main', graphics_mode=False):
        super(DynamicCar, self).__init__(x, y, width, length, angle, vel, acc, max_vel, mass, screen, screen_size, texture, graphics_mode)
        self.mass = 1000.0
        self.l_f = self.l_r = width / 2.0
        self.dangle = self.a_f = self.dx_body = self.dy_body = 0.0
        self.count = 0
        self.friction = 0.9

    def step(self, action, info_dict=None):
        self.count += 1
        delta_f, a_f = action

        # Convert to radians 
        delta_f, rad_angle, rad_dangle = np.radians(delta_f), np.radians(self.angle), np.radians(self.dangle)

        # Slip angles
        alpha_f = delta_f
        alpha_r = delta_f

        # Friction coefficient
        if info_dict is None:
            mu = 0.9
        else:
            collisions = info_dict['terrain_collisions']
            if len(collisions) == 0:
                mu = 0.9
            else:
                mu = min([terrain.friction for terrain in collisions])

        # Yaw Inertia
        I_z = 2510.15

        # Tire cornering stiffness
        c_f_est = self.mass * (self.l_r / (self.l_f + self.l_r))
        c_f = c_r = mu * c_f_est

        # Cornering force
        F_cf = -c_f * alpha_f
        F_cr = -c_r * alpha_r

        # Differential equations
        ddx_body = rad_dangle * self.dy_body + a_f
        ddy_body = -rad_dangle * self.dx_body + (2 / self.mass) * (F_cf * np.cos(delta_f) + F_cr)
        
        # Clamp acceleration if above maximum velocity
        body_vel = np.sqrt((ddx_body + self.dx_body) ** 2 + (ddy_body + self.dy_body) ** 2)
        if body_vel > self.max_vel:
            a = ddx_body ** 2 + ddy_body ** 2
            b = 2 * (ddx_body * self.dx_body + ddy_body * self.dy_body)
            c = self.dx_body ** 2 + self.dy_body ** 2 - self.max_vel ** 2
            sqrt_term = b**2 - 4*a*c

            # Truncate if ratio is too small to avoid floating point error
            epsilon = 0.0001
            if sqrt_term < epsilon:
                ratio = 0.0
            else:
                ratios = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a) , (-b - np.sqrt(b**2 - 4*a*c)) / (2*a) 
                ratio = max(ratios)
            ddx_body, ddy_body = ddx_body * ratio, ddy_body * ratio

        ddangle = (2 / I_z) * (self.l_f * F_cf - self.l_r * F_cr)
        dx = self.dx_body * np.cos(rad_angle) - self.dy_body * np.sin(rad_angle)
        dy = self.dx_body * np.sin(rad_angle) + self.dy_body * np.sin(rad_angle)

        # Clamp velocity
        vel = np.sqrt(dx ** 2 + dy ** 2)
        if vel > self.max_vel:
            ratio = self.max_vel / vel
            dx, dy = dx * ratio, dy * ratio
            
        # Update car 
        self.angle += np.rad2deg(rad_dangle + 0.5 * ddangle ** 2)
        self.angle %= 360.0
        self.dx_body = max(self.dx_body + ddx_body, 0.0)
        self.dy_body = max(self.dy_body + ddy_body, 0.0)
        self.x += dx
        self.y += dy
        self.dangle = np.rad2deg(delta_f)
        self.a_f = a_f
        self.body_vel = np.sqrt(self.dx_body ** 2 + self.dy_body ** 2)
        self.vel = np.sqrt(dx ** 2 + dy ** 2)

        # debug_list = ['action', 'dx', 'dy', 'self.vel', 'self.body_vel', 'self.dx_body', 'self.dy_body', \
        # 'ddx_body', 'ddy_body', 'ddangle', 'rad_dangle', 'a_f']
        # for item in debug_list:
        #     print(item, eval(item))
        # if self.count % 20 == 0:
        #     IPython.embed()