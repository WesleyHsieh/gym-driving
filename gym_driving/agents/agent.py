import numpy as np
import multiprocessing
from functools import partial
import copy
import IPython

class Agent(object):
    def __init__(self, learner, env):
        self.learner = learner
        self.env = env
