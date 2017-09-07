import numpy as np
from gym_driving.agents.agent import *
import IPython
import tensorflow as tf
import ray

class SupervisedAgent(Agent):

    def __init__(self, learner, env, supervisor):
        super(SupervisedAgent, self).__init__(learner, env)
        self.supervisor = supervisor
        self.rewards, self.surrogate_losses = [], []
        self.iterations = 0.0
        self.initialized = False

    def setup(self):
        print "Setting up the session...."
        self.learner.net.sess = tf.Session(graph=self.learner.net.g)
        self.learner.net.sess.run(self.learner.net.initializer)

        loss = self.learner.net.loss
        sess = self.learner.net.sess

        self.variables = ray.experimental.TensorFlowVariables(loss, sess)
        self.initialized = True

    def rollout_algorithm(self):
        """
        Rollout algorithm on underlying environment
        for one trajectory, using supervisor.
        """
        states, actions = [], []
        done = False
        state = self.env._reset()
        while not done:
            supervisor_label = self.supervisor.eval_policy(self.env)
            action_packed = [supervisor_label, 1]
            next_state, _, done, _ = self.env._step(action_packed)

            states.append(state)
            actions.append(supervisor_label)
            state = next_state

        return states, actions

    def eval_policy(self):
        """
        Evaluate underlying learner's policy.
        """
        states, actions, rewards, supervisor_labels, surrogate_losses = [], [], [], [], []
        done = False
        state = self.env._reset()
        while not done:
            supervisor_label = self.supervisor.eval_policy(self.env)
            action = self.learner.eval_policy(state)
            action_packed = [action, 1]
            next_state, reward, done, info_dict = self.env._step(action_packed)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            supervisor_labels.append(supervisor_label)
            surrogate_losses.append(action != supervisor_label)
            state = next_state

        self.rewards.append(sum(rewards))
        self.surrogate_losses.append(sum(surrogate_losses))

        return states, actions, self.rewards, supervisor_labels, self.surrogate_losses

    def update_model(self, states, actions):
        """
        Update model of underlying learner.
        """
        self.learner.add_to_data(states, actions)
        self.learner.train_learner()
        self.iterations += 1

    def get_statistics(self):
        """
        Train/test accuracy, reward, surrogate loss
        """
        train_loss, test_loss = self.learner.get_statistics()
        return train_loss, test_loss, np.mean(self.rewards), np.mean(self.surrogate_losses)

    def reset(self):
        self.learner.reset()
        self.iterations = 0
        self.rewards, self.surrogate_losses = [], []
