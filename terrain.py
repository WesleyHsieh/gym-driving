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
            self.texture_image = pygame.image.load('images/{}_tile_lite.jpg'.format(texture))
        else:
            print('Error: invalid texture')
        self.texture = texture
        self.decel = self.terrain_properties[texture]['decel']
        self.slip = self.terrain_properties[texture]['slip']
        self.screen = screen
        self.screen_size = screen_size
        self.tile_coords = []

        for i in range(-int(width / 2), int(width / 2), 100):
            for k in range(-int(length / 2), int(length / 2), 100):
                # Top left corners of each 100x100 tile
                self.tile_coords.append((x + i, y + k))

    def update_graphics(self, screen_coord):
        # Subtract screen_coord to get screen pos
        for coord in self.tile_coords:
            if coord[0] - screen_coord[0] < self.screen_size[0] and coord[1] - screen_coord[1] < self.screen_size[1]:
                pos = (int(coord[0] - screen_coord[0]), int(coord[1] - screen_coord[1]))
                self.screen.blit(self.texture_image, pos)
                pygame.draw.circle(self.screen, 0, pos, 5, 0)
