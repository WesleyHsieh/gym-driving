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

        for i in range(0, int(width/128)):
            for k in range(0, int(length/128)):
                # Centers of each 128x128 tile
                self.tile_coords.append((x + i*128, y + k*128))

    def update_graphics(self, screen_coord):
        # Subtract screen_coord to get screen pos, subtract 64/64 to get top left corner
        for coord in self.tile_coords:
            if coord[0] - screen_coord[0] - 64 < self.screen_size[0] and coord[1] - screen_coord[1] - 64 < self.screen_size[1]:
                pos = (int(coord[0] - screen_coord[0] - 64), int(coord[1] - screen_coord[1] - 64))
                self.screen.blit(self.texture_image, pos)
                pygame.draw.circle(self.screen, 0, pos, 5, 0)