import sys, os
import numpy as np
from numpy.random import rand,randint
import cv2
import IPython
from sklearn.cross_validation import train_test_split

class Learner(object):

    def compile_dataset(self, dataset='train', tensor=False):
        """
        Read into memory on request
        :param n: number of examples to return in batch
        :return: tuple with images in [0] and labels in [1]
        """
        if dataset == 'train':
            states = np.array([state for traj in self.train_states for state in traj])
            labels = np.array([label for traj in self.train_labels for label in traj])
        elif dataset == 'test':
            states = np.array([state for traj in self.test_states for state in traj])
            labels = np.array([label for traj in self.test_labels for label in traj])
        else:
            raise NotImplementedError
        if tensor:
            states = np.expand_dims(states, 1)
        return states, labels

    def add_to_data(self, states, labels):
        num_trajectories = len(states)
        train_states, test_states, train_labels, test_labels = train_test_split(states, labels, test_size=0.1)

        train_states_processed = [[self.preprocess_image(image) for image in traj] for traj in train_states]
        test_states_processed = [[self.preprocess_image(image) for image in traj] for traj in test_states]

        self.train_states.extend(train_states_processed)
        self.test_states.extend(test_states_processed)
        self.train_labels.extend(train_labels)
        self.test_labels.extend(test_labels)

    def downsample_image(self, state, n_iters=1):
        for _ in range(n_iters):
            w, h = state.shape
            state = cv2.pyrDown(state, dstsize = (w / 2, h / 2))
        return state

    def extract_HOG(self, state):
        winSize = (state.shape[0], state.shape[1])
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        # hog = cv2.HOGDescriptor()
        h = np.ravel(hog.compute(state))
        return h

    def reset(self):
        self.state_space = []
        self.labels = []

        self.train_states = []
        self.train_labels = []

        self.test_states = []
        self.test_labels = []

        self.test_loss = []
        self.train_loss = []
