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
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

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

        self.SAMPLES_PER_ROLLOUT = 3
        self.SAMPLES_PER_EVAL = 3
        self.ITERATIONS = 2
        self.TRIALS = 2

        self.file_path = file_path 

    def compute_averages(self, stats):
        train_average, test_average, reward_average, surr_loss_average = \
            [np.mean(np.array(stat)) for stat in zip(*stats)]
        return train_average, test_average, reward_average, surr_loss_average

    def save_data(self, stats, experiment_name):
        data_filepath = os.path.join(self.file_path, agent_name) + '.pkl'
        pickle.dump(stats, open(data_filepath,'wb'))
        self.plot_reward_curve(stats, experiment_name)

    def run_experiment(self, learner_name, agent_name):
        experiment_name = '{}_{}'.format(learner_name, agent_name)
        partial_func = partial(run_experiment_trial, learner_name=learner_name, agent_name=agent_name, \
            iterations=self.ITERATIONS, samples_per_rollout=self.SAMPLES_PER_ROLLOUT, samples_per_eval=self.SAMPLES_PER_EVAL)
        # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        # overall_stats = pool.map(partial_func, range(self.TRIALS))
        overall_stats = []
        for i in range(self.TRIALS):
            overall_stats.append(partial_func(i))
        # overall_stats = [partial_func(i) for i in range(self.TRIALS)]
        self.save_data(overall_stats, experiment_name)

    def plot_reward_curve(self, stats, experiment_name):
        stats_names = ['train_loss', 'test_loss', 'reward', 'surrogate_loss']
        means, stds = np.mean(stats, axis=0), np.std(stats, axis=0)
        n_iters, n_stats = means.shape
        for i in range(n_stats):
            plt.figure()
            stats_name = stats_names[i]
            stat_means, stat_stds = means[:, i], stds[:, i]
            plt.errorbar(range(len(stat_means)), stat_means, yerr=stat_stds)
            plt.title("Driving, Learner: {}, Stat: {}".format(experiment_name, stats_name))
            plt.xlabel("Number of Iterations")
            plt.ylabel(stats_name)
            plt.savefig('stats/stats_{}_{}.png'.format(experiment_name, stats_name))
        
def run_experiment_trial(trial_number, learner_name, agent_name, iterations, samples_per_rollout, samples_per_eval):
    np.random.seed(trial_number)
    num_processes = multiprocessing.cpu_count() - 1
    # Set up learner, agent
    if learner_name == 'linear_learner':
        learner = LinearLearner()
    elif learner_name == 'deep_learner':
        learner = DeepLearner()
    env_list = [DrivingEnv(graphics_mode=True) for _ in range(samples_per_rollout)]
    supervisor_list = [DrivingAgent() for _ in range(samples_per_rollout)]
    if agent_name == 'supervised':
        agent = SupervisedAgent(learner, env_list, supervisor_list)
    elif agent_name == 'dagger':
        agent = DaggerAgent(learner, env_list, supervisor_list)

    # Run trial
    trial_stats = []
    for j in range(iterations):
        print("Agent {}: Trial {}, Iteration {}".format(agent_name, trial_number, j))
        # Collect samples 
        state_list, action_list = agent.rollout_algorithm(samples_per_rollout)
        # Update model
        agent.update_model(state_list, action_list)
        # Evaluate policy
        agent.eval_policy(samples_per_eval)
        trial_stats.append(agent.get_statistics())
    print("Stats for Trial {}: ".format(trial_number))
    print("Train acc, Test acc, Reward, Surrogate Loss")
    print(trial_stats)
    return trial_stats

if __name__ == '__main__':
    FILEPATH = 'stats'
    # learners = ['linear_learner', 'deep_learner']
    learners = ['linear_learner']
    agents = ['dagger', 'supervised']
    # agents = ['supervised']
    if not os.path.exists(FILEPATH):
        os.makedirs(FILEPATH)
    for learner_name in learners:
        for agent_name in agents:
            print('Running experiment with {}'.format(learner_name))
            exp_class = Experiment(FILEPATH)
            start = time.time()
            exp_class.run_experiment(learner_name, agent_name)
            end = time.time()
            print("time", end - start)
