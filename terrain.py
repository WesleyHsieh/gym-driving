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
        if texture == 'road':
            self.texture = pygame.image.load('images/road_tile_lite.jpg')
        elif texture == 'grass':
            self.texture = pygame.image.load('images/grass_tile_lite.jpg')
        elif texture == 'patchy':
            self.texture = pygame.image.load('images/patchy_tile_lite.jpg')
        elif texture == 'dirt':
            self.texture = pygame.image.load('images/dirt_tile_lite.jpg')
        elif texture == 'ice':
            self.texture = pygame.image.load('images/ice_tile_lite.jpg')
        elif texture == 'icegrass':
            self.texture = pygame.image.load('images/icegrass_tile_lite.jpg')
        else:
            print('Error: invalid texture')
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
