from rectangle import *

import pygame
from pygame.locals import *

class Terrain(Rectangle):
    """
    Terrain for environment.
    :param width/length: int
        multiple of 128.
    :param texture: string
        'road' or 'grass'.
    :param screen: PyGame screen object
    """
    def __init__(self, x, y, width, length, texture, screen, screen_size, angle=0.0):
    	super(Terrain, self).__init__(x, y, width, length, angle)
        self.terrain_properties = {
            'road': {'decel': 0, 'slip': 0},
            'grass': {'decel': 1, 'slip': 0},
            'patchy': {'decel': 1, 'slip': 0},
            'dirt': {'decel': 2, 'slip': 0},
            'ice': {'decel': 0, 'slip': 1},
            'icegrass': {'decel': 1, 'slip': 1},
        }
        if texture in self.terrain_properties:
            self.texture = pygame.image.load('images/{}_tile_lite.jpg'.format(texture))
        else:
            print('Error: invalid texture')
        self.decel = self.terrain_properties[texture]['decel']
        self.slip = self.terrain_properties[texture]['slip']
        self.screen = screen
        self.screen_size = screen_size
        self.tile_coords = []

        for i in range(0, width/128):
            for k in range(0, length/128):
                self.tile_coords.append((x + i*128, y + k*128))

    def update_graphics(self, screen_coord):
        for coord in self.tile_coords:
            if coord[0] - screen_coord[0] < self.screen_size[0] and coord[1] - screen_coord[1] < self.screen_size[1]:
                self.screen.blit(self.texture, (coord[0] - screen_coord[0], coord[1] - screen_coord[1]))
