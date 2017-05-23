import numpy as np

class Rectangle(object):
    """
    Baseline rectangle class.
    """
    def __init__(self, x, y, width=50, length=25, angle=0.0):
        """
        Initializes rectangle object.

        Args:
            x: float, starting x position.
            y: float, starting y position.
            angle: float, starting angle of car in degrees.
        """
        self.x = x
        self.y = y
        self.angle = angle
        self.width = width
        self.length = length
        self.corners = self.calculate_corners()

    def get_pos(self):
        """
        Returns x, y coordinates.
        
        Returns:
            x: float, x position.
            y: float, y position. 
        """
        return self.x, self.y
        
    def get_corners(self):
        """
        Returns corners. 
        Should be called to access corners.
        
        Returns:
            corners: list, contains top right, bottom right, top left, bottom left 
                corners of rectangle.
        """
        return self.corners

    def calculate_corners(self):
        """
        Calculates corners of rectangle after
        applying rotations. 
        Should be called during updates.
        
        Returns:
            corners: list, contains top right, bottom right, top left, bottom left 
                corners of rectangle.
        """
        angle = np.radians(self.angle)
        corner_offsets = np.array([self.width / 2.0, self.length / 2.0])
        centers = np.array([self.x, self.y])
        signs = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
        corner_offsets = signs * corner_offsets
        rotation_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_corners = np.dot(corner_offsets, rotation_mat.T) + centers
        return rotated_corners

    def collide_rect(self, other_rect):
        """
        CChecks whether any point in
        the other rectangle is contained.
        
        Returns:
            has_collision: boolean, whether any point is contained in 
            the current rectangle.
        """
        corners = self.get_corners()
        other_corners = other_rect.get_corners()
        has_collision = any([self.contains_point(point) for point in other_corners]) or \
            any([other_rect.contains_point(point) for point in corners])
        return has_collision

    def contains_point(self, point):
        """
        Checks whether input point is contained
        in the rectangle, according to the formula
        described.
        http://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
        
        Args:
            point: 1x2 array, point to check. 
        Returns: 
            contains: boolean, whether the point is contained.
        """
        a, b, c, d = self.get_corners()
        AM, AB, AC = point - a, b - a, c - a
        c1 = 0 <= np.dot(AM, AB) <= np.dot(AB, AB)
        c2 = 0 <= np.dot(AM, AC) <= np.dot(AC, AC)
        contains = c1 and c2
        return contains

    def distance_to_rectangle(self, other_rect):
        """
        Calculate minimum distance between any pair
        of corners of current rectangle and an input rectangle.
        
        Args:
            other_rect: rectangle object, rectnagle to compare to.
        Returns: 
            min_dist: float, minimum distance.
        """
        corners = self.get_corners()
        other_corners = other_rect.get_corners()
        distances = []
        for c in corners:
            for oc in other_corners:
                distances.append(np.square(np.linalg.norm(c - oc)))
        min_dist = min(distances)
        return min_dist
