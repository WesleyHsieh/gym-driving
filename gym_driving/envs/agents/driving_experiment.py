''''
    Used to run experiments 

    Author: Michael Laskey 

'''


import sys, os

import IPython
# from deep_lfd.tensor import inputdata
# from compile_sup import Compile_Sup 
import numpy as np, argparse
# from deep_lfd.synthetic.affine_synthetic import Affine_Synthetic
import cPickle as pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deep_lfd.learning_driving.linear_learner import *
# from deep_lfd.learning_driving.deep_learner import *
from gym_driving.envs.agents.supervised_agent import *
from gym_driving.envs.agents.dagger_agent import *
from gym_driving.envs.agents.driving_agent import *
from gym_driving.envs.driving_env import *

os.environ["SDL_VIDEODRIVER"] = "dummy"

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
            [np.mean(np.array(stat)) for stat in zip(*stats)]
        return train_average, test_average, reward_average, surr_loss_average

    def save_data(self, stats, agent_name):
        # trn_avg,tst_avg,reward_avg,surr_loss_avg = self.compute_averages(stats)
        # print("Average train acc, test acc, reward, surrogate loss")
        # print(trn_avg,tst_avg,reward_avg,surr_loss_avg)
        data_filepath = os.path.join(self.file_path, agent_name) + '.pkl'
        pickle.dump(stats, open(data_filepath,'wb'))
        self.plot_reward_curve(stats, agent_name)

    def run_experiment(self, agent, agent_name):

        overall_stats = []

        for i in range(self.TRIALS):
            trial_stats = []
            for j in range(self.ITERATIONS):
                print("Agent {}: Trial {}, Iteration {}".format(agent_name, i, j))
                state_list, action_list = [], []
                # Collect Samples 
                # print("Rolling out algorithm")
                for k in range(self.SAMPLES_PER_ROLLOUT):   
                    states, actions = agent.rollout_algorithm()
                    state_list.append(states)
                    action_list.append(actions)
                # print("Updating Model")
                agent.update_model(state_list, action_list)
                # print("Evaluating Policy")
                for k in range(self.SAMPLES_PER_EVAL):
                    agent.eval_policy()
                trial_stats.append(agent.get_statistics())
            overall_stats.append(trial_stats)
            print("Stats for Trial {}: ".format(i))
            print("Average train acc, test acc, reward, surrogate loss")
            print(overall_stats[i])
            agent.reset()
        self.save_data(overall_stats, agent_name)

    def plot_reward_curve(self, stats, agent_name):
        # stats: (trials, iterations, stats)
        # means, stds: (iterations, stats)
        stats_names = ['train_loss', 'test_loss', 'reward', 'surrogate_loss']
        means, stds = np.mean(stats, axis=0), np.std(stats, axis=0)
        n_iters, n_stats = means.shape
        for i in range(n_stats):
            plt.figure()
            stats_name = stats_names[i]
            stat_means, stat_stds = means[:, i], stds[:, i]
            plt.errorbar(range(len(stat_means)), stat_means, yerr=stat_stds)
            plt.title("Driving, Learner: {}, Stat: {}".format(agent_name, stats_name))
            plt.xlabel("Number of Iterations")
            plt.ylabel(stats_name)
            plt.savefig('stats/stats_{}_{}.png'.format(agent_name, stats_name))
        
if __name__ == '__main__':
    FILEPATH = ''
    # learners = [('linear_learner', LinearLearner()), ('deep_learner', DeepLearner())]
    learners = ['linear_learner']
    agents = ['supervised', 'dagger']
    for learner_name in learners:
        if learner_name == 'linear_learner':
            learner = LinearLearner()
        elif learner_name == 'deep_learner':
            learner = DeepLearner()
        for agent_name in agents:
            # learner = LinearLearner()
            # learner = DeepLearner()
            start = time.time()
            print('Running experiment with {}'.format(learner_name))
            env = DrivingEnv(graphics_mode=True)
            supervisor = DrivingAgent()
            if agent_name == 'supervised':
                agent = SupervisedAgent(learner, env, supervisor)
            elif agent_name == 'dagger':
                agent = DaggerAgent(learner, env, supervisor)
            exp_class = Experiment(FILEPATH)
            experiment_name = '{}_{}'.format(learner_name, agent_name)
            exp_class.run_experiment(agent, experiment_name)
            end = time.time()
            print("Time Elapsed: ", end - start)

