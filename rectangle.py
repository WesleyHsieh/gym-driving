import numpy as np

class Rectangle(object):
    def __init__(self, x, y, width=50, length=25, angle=0.0):
        self.x = x
        self.y = y
        self.angle = angle
        self.width = width
        self.length = length

    def get_corners(self):
        """"
        Returns corners of rectangle. 
        :return: None
        """
        angle = self.angle
        corner_offsets = np.array([self.width / 2.0, self.length / 2.0])
        centers = np.array([self.x, self.y])
        signs = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
        corners = centers + signs * corner_offsets
        rotation_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_corners = np.dot(corners, rotation_mat.T)
        return rotated_corners

    def collide_rect(self, other_rect):
        """
        Checks whether any point in
        the other rectangle is contained.
        :return: boolean
            Whether any point is contained in 
            the current rectangle.
        """
        corners = self.get_corners()
        other_corners = other_rect.get_corners()
        return any([self.contains_point(point) for point in other_corners]) or \
            any([other_rect.contains_point(point) for point in corners])

    def contains_point(self, point):
        """
        Checks whether input point is contained
        in the rectangle, according to the formula
        described.
        http://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
        :return: boolean
            Whether the point is contained.
        """
        a, b, c, d = self.get_corners()
        AM, AB, AC = point - a, b - a, c - a
        c1 = 0 <= np.dot(AM, AB) <= np.dot(AB, AB) 
        c2 = 0 <= np.dot(AM, AC) <= np.dot(AC, AC)
        return c1 and c2

    def update_graphics(self):
        """
        TODO: Updates graphics.
        """
        pass