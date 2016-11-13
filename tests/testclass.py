import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from car import *
from environment import *
from simulator import *
from terrain import *
from rectangle import *
import IPython

class TestRectangle:
	def __init__(self):
		pass

	def test_get_corners(self):
		rect = Rectangle(x=0.0, y=0.0, width=10.0, length=20.0, angle=0.0)
		corners = rect.get_corners()
		assert np.array_equal(corners, 
			np.array([[5,10], [5,-10], [-5,10], [-5,-10]]))

		rect = Rectangle(x=0.0, y=0.0, width=10.0, length=20.0, angle=np.pi/4)
		angle = np.pi/4
		original_corners = np.array([[5,10], [5,-10], [-5,10], [-5,-10]])
		rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
		corners = rect.get_corners()
		rot_corners = np.dot(original_corners, rot_mat.T)
		assert np.array_equal(corners, rot_corners)

	def test_contains_point(self):
		rect = Rectangle(x=0.0, y=0.0, width=10.0, length=20.0, angle=0.0)
		true_point = np.array([5, 5])
		false_point = np.array([5.01, 5.01])
		assert rect.contains_point(true_point)
		assert not rect.contains_point(false_point)

	def test_collide_rect(self):
		rect = Rectangle(x=0.0, y=0.0, width=10.0, length=20.0, angle=0.0)
		rect_true = Rectangle(x=2.5, y=2.5, width=10.0, length=20.0, angle=0.0)
		rect_false = Rectangle(x=10.0, y=10.0, width=1.0, length=1.0, angle=np.pi/4)
		assert rect.collide_rect(rect_true)
		assert not rect.collide_rect(rect_false)
