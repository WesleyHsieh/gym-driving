import numpy as np
import multiprocessing
from functools import partial
import copy
import IPython

class Agent(object):
	def __init__(self, learner, env_list):
		self.learner = learner
		self.env_list = env_list