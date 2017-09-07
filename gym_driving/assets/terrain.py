from gym_driving.assets.rectangle import *

import pygame
import numpy as np
from pygame.locals import *
import os

class Terrain:
    """
    Wrapper class that uses RectangleTerrain() if the terrain type is rectangular, and CircularArcTerrain if the type is
    circular.
    """
    def __init__(self, x, y, width, length, texture, screen, screen_size, angle=0.0, render_mode=False):
        self.x, self.y, self.width, self.length = x, y, width, length
        self.texture, self.screen, self.screen_size = texture, screen, screen_size
        self.angle, self.render_mode = angle, render_mode

    def create(self):
        """
        Used to create the terrain object which the simulator interacts with.
        """
        if self.angle == 0.0:
            return RectangularTerrain(self.x, self.y, self.width, self.length, self.texture, self.screen, \
                self.screen_size, angle=self.angle, render_mode=self.render_mode)
        else:
            return RotatableTerrain(self.x, self.y, self.width, self.length, self.texture, self.screen, \
                self.screen_size, angle=self.angle, render_mode=self.render_mode)



class RectangularTerrain(Rectangle):
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
        super(RectangularTerrain, self).__init__(x, y, width, length, angle)
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
                filename = os.path.join(base_dir, 'sprites', '{}_tile_lite.jpg'.format(texture))
                self.texture_image = pygame.image.load(filename)
            else:
                print('Error: invalid terrain texture')
        self.texture = texture
        self.friction = self.terrain_properties[texture]['friction']
        self.screen = screen
        self.screen_size = screen_size
        self.tile_coords = []
        if angle == 0.0:
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



class RotatableTerrain(RectangularTerrain):
    """
    Road which goes in a curve, defined by a point, radius, angle, and offset from 0 degrees.
    """
    def __init__(self, x, y, width, length, texture, screen, screen_size, angle, render_mode=False):
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
        super(RotatableTerrain, self).__init__(x, y, width, length, texture, screen, screen_size, angle, render_mode)
        # Tile coords need to be changed from original
        self.tile_coords = []

        # Create rotated tile coordinates for rendering
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        self.rotation_matrix = np.matrix([[c, -s], [s, c]])
        for i in range(-int(width / 2), int(width / 2), 100):
            for k in range(-int(length / 2), int(length / 2), 100):
                original = np.array([i, k])
                rotated = np.matmul(self.rotation_matrix, original).tolist()[0]

                self.tile_coords.append((rotated[0]+x, rotated[1]+y))

    def render(self, screen_coord):
        """
        Renders car.

        Args:
            screen_coord: 1x2 array, coordinates of center of screen.
        """
        assert self.render_mode is True

        image_rotated = pygame.transform.rotate(self.texture_image, -self.angle)
        for coord in self.tile_coords:
            #corners, center, angle = coord.get_corners(), coord.get_pos(), coord.angle
            #x_offset = (np.abs((coord.width - coord.length) * np.cos(np.radians(angle))) + coord.length) / 2
            #y_offset = (np.abs((coord.width - coord.length) * np.sin(np.radians(angle))) + coord.length) / 2
            # Subtract screen_coord to get screen pos
            if -100 <= coord[0] - screen_coord[0] <= self.screen_size[0] and -100 <= coord[1] - screen_coord[1] <= self.screen_size[1]:
                #pos = (int(center[0] - screen_coord[0] - x_offset), int(center[1] - screen_coord[1] - y_offset))
                pos = (int(coord[0] - screen_coord[0]), int(coord[1] - screen_coord[1]))
                self.screen.blit(image_rotated, pos)
