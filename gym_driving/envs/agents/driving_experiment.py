''''
    Used to run experiments 

    Author: Michael Laskey 

'''


import sys, os

import IPython
from deep_lfd.tensor import inputdata
# from compile_sup import Compile_Sup 
import numpy as np, argparse
# from deep_lfd.synthetic.affine_synthetic import Affine_Synthetic
import cPickle as pickle

from deep_lfd.learning_driving.linear_learner import *
from deep_lfd.learning_driving.deep_learner import *
from gym_driving.envs.agents.supervised_agent import *
from gym_driving.envs.agents.driving_agent import *
from gym_driving.envs.driving_env import *

class Experiment():

    def __init__(self, file_path):
        # self.SAMPLES_PER_ROLLOUT = 20
        # self.SAMPLES_PER_EVAL = 20
        # self.ITERATIONS = 20
        # self.TRIALS = 20

        self.SAMPLES_PER_ROLLOUT = 2
        self.SAMPLES_PER_EVAL = 2
        self.ITERATIONS = 2
        self.TRIALS = 2

        self.file_path = file_path 


    def compute_averages(self, stats):
        train_average, test_average, reward_average, surr_loss_average = \
            [np.mean(stat) for stat in zip(*stats)]
        return train_average, test_average, reward_average, surr_loss_average

    def save_data(self, stats, agent_name):

        trn_avg,tst_avg,reward_avg,surr_loss_avg = self.compute_averages(stats)
        print("Average train acc, test acc, reward, surrogate loss")
        print(trn_avg,tst_avg,reward_avg,surr_loss_avg)
        pickle.dump(stats,open(self.file_path+agent_name,'wb'))

    def run_experiment(self, agent):

        stats = []

        for i in range(self.TRIALS):
            for j in range(self.ITERATIONS):
                state_list, action_list = [], []
                # Collect Samples 
                for k in range(self.SAMPLES_PER_ROLLOUT):
                    print("Rolling out algorithm")
                    states, actions = agent.rollout_algorithm()
                    state_list.append(states)
                    action_list.append(actions)
                print("Updating Model")
                agent.update_model(state_list, action_list)
                for k in range(self.SAMPLES_PER_EVAL):
                    print("Evaluating Policy")
                    agent.eval_policy()
            stats.append(agent.get_statistics())
            agent.reset_statistics()

        self.save_data(stats, 'driving_agent_stats')
        
if __name__ == '__main__':
    FILEPATH = ''
    AGENT_NAME = ''

    # learner = LinearLearner()
    learner = DeepLearner()
    env = DrivingEnv(graphics_mode=True)
    supervisor = DrivingAgent()
    agent = SupervisedAgent(learner, env, supervisor)

    exp_class = Experiment(FILEPATH)
    exp_class.run_experiment(agent)