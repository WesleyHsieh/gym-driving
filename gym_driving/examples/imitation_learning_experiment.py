import sys, os

import IPython
import numpy as np, argparse
import cPickle as pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

from gym_driving.models.linear_learner import *
from gym_driving.models.deep_learner import *
from gym_driving.agents.supervised_agent import *
from gym_driving.agents.dagger_agent import *
from gym_driving.agents.search_agent import *
from gym_driving.envs.driving_env import *

os.environ["SDL_VIDEODRIVER"] = "dummy"

class Experiment():

    def __init__(self, file_path):
        # self.SAMPLES_PER_ROLLOUT = 20
        # self.SAMPLES_PER_EVAL = 20
        # self.ITERATIONS = 20
        # self.TRIALS = 10

        self.SAMPLES_PER_ROLLOUT = 2
        self.SAMPLES_PER_EVAL = 2
        self.ITERATIONS = 2
        self.TRIALS = 2

        self.file_path = file_path

    def compute_averages(self, stats):
        train_average, test_average, reward_average, surr_loss_average = \
            [np.mean(np.array(stat)) for stat in zip(*stats)]
        return train_average, test_average, reward_average, surr_loss_average

    def save_data(self, stats, experiment_name):
        data_filepath = os.path.join(self.file_path, experiment_name) + '.pkl'
        pickle.dump(stats, open(data_filepath,'wb'))
        self.plot_reward_curve(stats, experiment_name)

    def run_experiment(self, learner_name, agent_name):
        experiment_name = '{}_{}'.format(learner_name, agent_name)
        partial_func = partial(run_experiment_trial, learner_name=learner_name, agent_name=agent_name, \
            iterations=self.ITERATIONS, samples_per_rollout=self.SAMPLES_PER_ROLLOUT, samples_per_eval=self.SAMPLES_PER_EVAL)
        overall_stats = [partial_func(i) for i in range(self.TRIALS)]
        self.save_data(overall_stats, experiment_name)

    def plot_reward_curve(self, stats, experiment_name):
        plot_coords = [221, 222, 223, 224]
        stats_names = ['train_loss', 'test_loss', 'reward', 'surrogate_loss']
        means, stds = np.mean(stats, axis=0), np.std(stats, axis=0)
        n_iters, n_stats = means.shape
        plt.figure()
        for i in range(n_stats):
            plt.subplot(plot_coords[i])
            stats_name = stats_names[i]
            stat_means, stat_stds = means[:, i], stds[:, i]
            plt.errorbar(range(len(stat_means)), stat_means, yerr=stat_stds)
            if i == 2:
                plt.plot(range(len(stat_means)), [79.137692596025317 for _ in range(len(stat_means))])
            plt.title(stats_name)
            # plt.title("Driving, Learner: {}, Stat: {}".format(experiment_name, stats_name))
            # plt.xlabel("Number of Iterations")
            # plt.ylabel(stats_name)
        plt.savefig('stats/stats_{}.png'.format(experiment_name))

def run_experiment_trial(trial_number, learner_name, agent_name, iterations, samples_per_rollout, samples_per_eval):
    np.random.seed(trial_number)
    # Set up learner, agent
    if learner_name == 'linear_learner':
        learner = LinearLearner()
    elif learner_name == 'deep_learner':
        learner = DeepLearner()

    config_filepath = '../configs/driving_experiment_config.json'
    env = DrivingEnv(render_mode=True, config_filepath=config_filepath)
    supervisor = SearchAgent(env=env)
    if agent_name == 'supervised':
        agent = SupervisedAgent(learner, env, supervisor)
    elif agent_name == 'dagger':
        agent = DaggerAgent(learner, env, supervisor)

    # Run trial
    trial_stats = []
    for j in range(iterations):
        print("Agent {}: Trial {}, Iteration {}".format(agent_name, trial_number, j))
        # Collect samples
        state_list, action_list = [], []
        for _ in range(samples_per_rollout):
            states, actions = agent.rollout_algorithm()
            state_list.append(states)
            action_list.append(actions)
        # Update model
        agent.update_model(state_list, action_list)
        # Evaluate policy
        for _ in range(samples_per_eval):
            agent.eval_policy()
        trial_stats.append(agent.get_statistics())
    print("Stats for Trial {}: ".format(trial_number))
    print("Train acc, Test acc, Reward, Surrogate Loss")
    print(trial_stats)
    return trial_stats

if __name__ == '__main__':
    FILEPATH = 'stats'
    learners = ['linear_learner', 'deep_learner']
    agents = ['dagger', 'supervised']
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
