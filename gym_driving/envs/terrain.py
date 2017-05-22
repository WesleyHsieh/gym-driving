from gym_driving.envs.rectangle import *

import pygame
from pygame.locals import *
import os

class Terrain(Rectangle):
    """
    Terrain for environment.
    """
    def __init__(self, x, y, width, length, texture, screen, screen_size, angle=0.0, render_mode=False):
        """
        Initializes terrain object.

        Args:
            x: float, starting x position.
            y: float, starting y position.
            width: int, width of terrain.
            length: int, length of terrain.
            texture: str, texture of terrain for rendering, 
                must be one of the options in textures.
            screen: PyGame screen object, used for rendering.
            screen_size: 1x2 array, size of screen in pixels.
            angle: float, angle of object in degrees.
            render_mode: boolean, whether to render.
        """
        super(Terrain, self).__init__(x, y, width, length, angle)
        self.terrain_properties = {
            'road': {'friction': 0.9},
            'grass': {'friction': 0.6},
            'patchy': {'friction': 0.9},
            'dirt': {'friction': 0.9},
            'ice': {'friction': 0.2},
            'icegrass': {'friction': 0.6},
        }
        self.render_mode = render_mode
        if self.render_mode:
            if texture in self.terrain_properties:
                base_dir = os.path.dirname(__file__)
                filename = os.path.join(base_dir, 'images', '{}_tile_lite.jpg'.format(texture))
                self.texture_image = pygame.image.load(filename)
            else:
                print('Error: invalid terrain texture')
        self.texture = texture
        self.friction = self.terrain_properties[texture]['friction']
        self.screen = screen
        self.screen_size = screen_size
        self.tile_coords = []

        for i in range(-int(width / 2), int(width / 2), 100):
            for k in range(-int(length / 2), int(length / 2), 100):
                # Top left corners of each 100x100 tile
                self.tile_coords.append((x + i, y + k))

    def render(self, screen_coord):
        """
        Renders terrain.

        Args:
            screen_coord: 1x2 array, coordinates of center of screen
        """
        assert self.render_mode is True
        # Subtract screen_coord to get screen pos
        for coord in self.tile_coords:
            if -100 <= coord[0] - screen_coord[0] <= self.screen_size[0] and -100 <= coord[1] - screen_coord[1] <= self.screen_size[1]:
                pos = (int(coord[0] - screen_coord[0]), int(coord[1] - screen_coord[1]))
                self.screen.blit(self.texture_image, pos)
                #pygame.draw.circle(self.screen, 0, pos, 5, 0)
