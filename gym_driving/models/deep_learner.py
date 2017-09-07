import sys, os

import IPython
import numpy as np, argparse
import cv2
from numpy.random import rand,randint

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical

from gym_driving.models.learner import *

class DeepLearner(Learner):
    def __init__(self):
        super(DeepLearner, self).__init__()
        model = Sequential()
        model.add(Convolution2D(5, 7, 7, border_mode='same', input_shape=(1, 128, 128)))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(units=60))
        model.add(Activation("relu"))
        model.add(Dense(units=5))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.net = model

        # Use global graph to avoid Tensorflow errors
        global graph
        graph = tf.get_default_graph()

        self.init_weights = self.get_weights()
        self.reset()

    def train_learner(self):
        global graph
        with graph.as_default():
            train_states, train_labels = self.compile_dataset('train', tensor=True)
            train_labels = to_categorical(train_labels, num_classes=5)
            self.net.fit(train_states, train_labels, nb_epoch=5, batch_size=32, verbose=0)

    def eval_policy(self, state):
        global graph
        with graph.as_default():
            processed_state = self.preprocess_image(state)
            state_expanded = np.expand_dims(np.expand_dims(processed_state, 0), 1)
            output = self.net.predict_classes(state_expanded, verbose=0)[0]
            return output

    def get_statistics(self):
        global graph
        with graph.as_default():
            train_states, train_labels = self.compile_dataset('train', tensor=True)
            train_labels = to_categorical(train_labels, num_classes=5)
            train_loss, train_acc = self.net.evaluate(train_states, train_labels, verbose=0)
            test_states, test_labels = self.compile_dataset('test', tensor=True)
            test_labels = to_categorical(test_labels, num_classes=5)
            if len(test_states) > 0:
                test_loss, test_acc = self.net.evaluate(test_states, test_labels, verbose=0)
            else:
                test_loss, test_acc = [np.inf, 0.0]
            return train_acc, test_acc

    def preprocess_image(self, state):
        downsampled = self.downsample_image(self.downsample_image(state))
        return downsampled

    def reset(self):
        super(DeepLearner, self).reset()
        init_weights = self.get_weights()
        self.set_weights(init_weights)

    def get_weights(self):
        global graph
        with graph.as_default():
            return self.net.get_weights()

    def set_weights(self, weights):
        global graph
        with graph.as_default():
            self.net.set_weights(weights)

# class DeepLearner(Learner):
#     # TODO: Replace 'to_categorical' with one-hot vector
#     def __init__(self, gpu=False):
#         super(DeepLearner, self).__init__()
#         # traceback.print_stack()
#         os.environ["CUDA_VISIBLE_DEVICES"] = ""
#         if gpu:
#             os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#         self.net = Net_Driving(channels=1) #, on_gpu=gpu)
#         self.reset()

#     def train_learner(self,iterations):
#         print "SIZE OF TRAIN DATA ",len(self.train_states)
#         print "SIZE OF TEST DATA ", len(self.test_states)
#         data = IMData(self.train_states,self.train_labels,self.test_states,self.test_labels,channels=1)
#         if(iterations == 0):
#             initialize = True
#         else:
#             initialize = False
#         self.net.optimize(2000,data, batch_size=200,save=False,initialize = initialize)
#         #train_states, train_labels = self.compile_dataset('train', tensor=True)

#         # TODO: Fit model to training set
#         #self.net.fit(train_states, train_labels, nb_epoch=5, batch_size=32, verbose=0)

#     def eval_policy(self, state):
#         processed_state = self.preprocess_image(state)
#         #IPython.embed()
#         A = np.zeros([1,processed_state.shape[0],processed_state.shape[1],1])
#         A[0,:,:,0] = processed_state

#         # TODO: Predict output of net
#         output = self.net.predict(A/255.0)
#         return output

#     def get_statistics(self):
#         return self.net.train_loss, self.net.test_loss

#     def preprocess_image(self, state):

#         h,w = state.shape

#         size = 150

#         crop_image = state[h/2-size:h/2+size,w/2-size:size+w/2]

#         # cv2.imshow('debug',crop_image)
#         # cv2.waitKey(30)


#         return crop_image

#     def reset(self):
#         #TODO: Compile model
#         self.state_space = []
#         self.labels = []

#         self.train_states = []
#         self.train_labels = []

#         self.test_states = []
#         self.test_labels = []

#         self.test_loss = []
#         self.train_loss = []


#         #self.net = Net_Driving(channels=1)
