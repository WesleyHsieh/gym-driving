import sys, os

import IPython
import numpy as np

from sklearn import svm, multioutput, linear_model, ensemble
from gym_driving.models.learner import *

class LinearLearner(Learner):

    def __init__(self):
        super(LinearLearner, self).__init__()
        self.reset()

    def train_learner(self):
        X_train, Y_train = self.compile_dataset('train')
        self.net.fit(X_train, Y_train)

    def eval_policy(self, state):
        processed_state = self.preprocess_image(state).reshape(1, -1)
        output = self.net.predict(processed_state)[0]
        return output

    def get_statistics(self):
        train_states, train_labels = self.compile_dataset('train')
        train_acc = self.net.score(train_states, train_labels)
        test_states, test_labels = self.compile_dataset('test')
        if len(test_states) > 0:
            test_acc = self.net.score(test_states, test_labels)
        else:
            test_acc = 0.0
        return train_acc, test_acc

    def preprocess_image(self, state):
        squeezed = np.squeeze(state)
        downsampled = self.downsample_image(squeezed, n_iters=3)
        hog = self.extract_HOG(downsampled)
        return hog

    def reset(self):
        super(LinearLearner, self).reset()
        self.net = ensemble.RandomForestClassifier()
