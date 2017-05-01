import gym
import numpy as np
import tensorflow as tf
import time
import ray
import IPython
from ray_dart_agent import RayDartAgent
from ray_off_d_agent import RayOffAgent
from ray_dagger_agent import RayDaggerAgent

import cPickle as pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os

# from deep_lfd.learning_driving.linear_learner import *
from deep_lfd.learning_driving.deep_learner import *
from gym_driving.envs.agents.supervised_agent import *
from gym_driving.envs.agents.dagger_agent import *
from gym_driving.envs.agents.driving_agent import *
from gym_driving.envs.driving_env import *

agent_name = "test"
NUM_WORKERS = 4
os.environ["SDL_VIDEODRIVER"] = "dummy"

ray.init(num_workers=NUM_WORKERS)


def plot_reward_curve( stats, agent_name):
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







def env_init():
    return DrivingEnv(graphics_mode=False)

def env_reinit(env):
    return env

ray.env.env = ray.EnvironmentVariable(env_init, env_reinit)

def supervisor_init():
    return DrivingAgent()

def supervisor_reinit(spvsr):
    return spvsr

ray.env.supervisor = ray.EnvironmentVariable(supervisor_init, supervisor_reinit)

def agent_dart_init():
    env = ray.env.env
    supervisor = ray.env.supervisor
    return RayDartAgent(DeepLearner(), env, supervisor)



def agent_dagger_init():
    env = ray.env.env
    supervisor = ray.env.supervisor
    return RayDaggerAgent(DeepLearner(), env, supervisor)


def agent_off_init():
    env = ray.env.env
    supervisor = ray.env.supervisor
    return RayOffAgent(DeepLearner(), env, supervisor)

def agent_reinit(agent):
    agent.reset()
    return agent

#ray.env.off_agent = ray.EnvironmentVariable(agent_off_init, agent_reinit)
#ray.env.dart_agent = ray.EnvironmentVariable(agent_dart_init, agent_reinit)
ray.env.dagger_agent = ray.EnvironmentVariable(agent_dagger_init, agent_reinit)





@ray.remote
def rollout(weights, params,alg_type):
    print("## Starting Rollout...")
    if(alg_type == 'dagger'):
        agent = ray.env.dagger_agent
    elif(alg_type == 'off_d'):
        agent = ray.env.off_agent
    elif(alg_type == 'dart'):
        agent = ray.env.dart_agent
    #agent = ray.env.agent
    agent.set_weights(weights)
    agent.set_params(params)
    return agent.rollout_algorithm()


def save_data(stats, alg_name):
    data_filepath = os.path.join('results/', alg_name) + '.p'
    
    pickle.dump(stats, open(data_filepath,'wb'))
    plot_reward_curve(stats, alg_name)

def train(alg_type):
    TRIALS = 10
    ITERATIONS = 4
    SAMPLES_PER_ROLLOUT = 150
    SAMPLES_PER_EVAL = 20
    overall_stats = []
    

    if(alg_type == 'dagger'):

        main_agent = ray.env.dagger_agent
    elif(alg_type == 'off_d'):
        main_agent = ray.env.off_agent
    elif(alg_type == 'dart'):
        main_agent = ray.env.dart_agent

    for i in range(TRIALS):
        trial_stats = []

        for j in range(ITERATIONS):
            print("Agent {}: Trial {}, Iteration {}".format(agent_name, i, j))
            weights = main_agent.get_weights()
            weight_id = ray.put(weights)
            
            if(alg_type == 'dagger'):
                params = [j]
            elif(alg_type == 'dart'):
                params = [0.1]#[main_agent.compute_eps()]
            elif(alg_type == 'off_d'):
                params = [1]

            rollouts = [rollout.remote(weight_id, params,alg_type) for k in range(SAMPLES_PER_ROLLOUT)]
            results = ray.get(rollouts)
            print "Collected Rollouts"
            state_list, action_list = zip(*results)
            #IPython.embed()
            main_agent.update_model(state_list, action_list)

           

            print "EVAL MODEL"
            for k in range(SAMPLES_PER_EVAL):
                main_agent.eval_policy()
            trial_stats.append(main_agent.get_statistics())
            print "GET STATISTICS MAY NOT BE COMPLETELY CORRECT"
        overall_stats.append(trial_stats)
        print("Stats for Trial {}: ".format(i))
        print("Average train acc, test acc, reward, surrogate loss")
        print(overall_stats[i])
        main_agent.reset()
    save_data(overall_stats,alg_type)

if __name__ == '__main__':

    #train('off_d')

    train('dagger')

    #train('dart')
