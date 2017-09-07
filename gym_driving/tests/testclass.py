import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.examples.run_simulator import *
from gym_driving.assets.terrain import *
from gym_driving.assets.rectangle import *
import IPython

class TestRectangle:
    def __init__(self):
        pass

    def test_get_corners(self):
        rect = Rectangle(x=0.0, y=0.0, width=10.0, length=20.0, angle=0.0)
        corners = rect.get_corners()
        assert np.array_equal(corners,
            np.array([[5,10], [5,-10], [-5,10], [-5,-10]]))

        rect = Rectangle(x=0.0, y=0.0, width=10.0, length=20.0, angle=45.0)
        angle = np.radians(45.0)
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
        rect_false = Rectangle(x=10.0, y=10.0, width=1.0, length=1.0, angle=45.0)
        assert rect.collide_rect(rect_true)
        assert not rect.collide_rect(rect_false)

class TestEnvironment:
    def __init__(self):
        pass

    def test_get_state(self):
        env = Environment(screen_size=[100.0, 200.0], terrain=[], num_cpu_cars=0)
        state_dict, done = env.get_state()
        assert state_dict['main_car']['x'] == 50.0
        assert state_dict['main_car']['y'] == 100.0
        assert state_dict['main_car']['vel'] == 0.0
        assert state_dict['main_car']['angle'] == state_dict['main_car']['vel_angle'] == 0.0
        assert done is False

    def test_get_action(self):
        env = Environment(screen_size=[100.0, 200.0], terrain=[], num_cpu_cars=0)
        state_dict, done = env.get_state()
        assert state_dict['main_car']['x'] == 50.0
        assert state_dict['main_car']['y'] == 100.0
        assert state_dict['main_car']['vel'] == 0.0
        assert state_dict['main_car']['angle'] == state_dict['main_car']['vel_angle'] == 0.0
        assert done is False

        # Test acceleration
        action_dict = {'acc': 10.0, 'steer': 0.0}
        state_dict, reward, done = env.take_action(action_dict)
        assert state_dict['main_car']['x'] == 50.0 + 0.5 * 10.0 # 55.0
        assert state_dict['main_car']['y'] == 100.0
        assert state_dict['main_car']['vel'] == 10.0
        assert state_dict['main_car']['angle'] == state_dict['main_car']['vel_angle'] == 0.0

        # Test braking
        action_dict = {'acc': -10.0, 'steer': 0.0}
        state_dict, reward, done = env.take_action(action_dict)
        assert state_dict['main_car']['x'] == 55.0 + 10.0 - 0.5 * 10.0 # 60.0
        assert state_dict['main_car']['y'] == 100.0
        assert state_dict['main_car']['vel'] == 0.0
        assert state_dict['main_car']['angle'] == state_dict['main_car']['vel_angle'] == 0.0

        # Test steering
        action_dict = {'acc': 0.0, 'steer': 45.0}
        state_dict, reward, done = env.take_action(action_dict)
        assert state_dict['main_car']['x'] == 60.0
        assert state_dict['main_car']['y'] == 100.0
        assert state_dict['main_car']['vel'] == 0.0
        assert state_dict['main_car']['angle'] == state_dict['main_car']['vel_angle'] == 45.0

        action_dict = {'acc': 0.0, 'steer': -45.0}
        state_dict, reward, done = env.take_action(action_dict)
        assert state_dict['main_car']['x'] == 60.0
        assert state_dict['main_car']['y'] == 100.0
        assert state_dict['main_car']['vel'] == 0.0
        assert state_dict['main_car']['angle'] == state_dict['main_car']['vel_angle'] == 0.0

        action_dict = {'acc': 0.0, 'steer': -45.0}
        state_dict, reward, done = env.take_action(action_dict)
        assert state_dict['main_car']['x'] == 60.0
        assert state_dict['main_car']['y'] == 100.0
        assert state_dict['main_car']['vel'] == 0.0
        assert state_dict['main_car']['angle'] == state_dict['main_car']['vel_angle'] == 315.0

    def test_terrain(self):
        SCREEN_SIZE = (1000.0, 1000.0)
        terrain = Terrain(x=50.0, y=200.0, width=200.0, length=400.0, texture='road',
            screen=None, screen_size=SCREEN_SIZE)
        terrains = [terrain]
        env = Environment(screen_size=SCREEN_SIZE, terrain=terrains, num_cpu_cars=0)

        # width = 50, length = 25
        in_car = Car(x=100.0, y=100.0)
        env.main_car = in_car
        state_dict, _ = env.get_state()
        assert state_dict['terrain_collisions'][0] == True

        out_car = Car(x=100., y=100.0)
        env.main_car = in_car
        state_dict, _ = env.get_state()
        assert state_dict['terrain_collisions'][0] == False
