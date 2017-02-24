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

ray.init(num_workers=NUM_WORKERS, num_cpus=4)

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

        plt.savefig('stats2/stats_{}_{}.png'.format(agent_name, stats_name))







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
    print "Agent..."
    a =  RayDartAgent(DeepLearner(), env, supervisor)
    print "Done agent"
    return a


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
ray.env.dart_agent = ray.EnvironmentVariable(agent_dart_init, agent_reinit)
#ray.env.dagger_agent = ray.EnvironmentVariable(agent_dagger_init, agent_reinit)



@ray.remote
def rollout(weights, params,alg_type, c=0):
    #"""
    if(alg_type == 'dagger'):
        agent = ray.env.dagger_agent
    elif(alg_type == 'off_d'):
        agent = ray.env.off_agent
    elif(alg_type == 'dart_off'):
        agent = ray.env.dart_agent


    if not agent.initialized:
        agent.setup()
        print "Setting up agent"
    agent.set_weights(weights)
    agent.set_params(params)
    print("## Starting Rollout %d..." % c)
    for i in range(5):
        x = agent.rollout_algorithm()
    print("## Finished Rollout %d" % c)
    return [x] * 10
    #"""
    #import time
    #time.sleep(1)
    #return np.zeros([100, 300, 300]), np.zeros([100, 5], dtype=int)


def save_data(stats, alg_name):
    data_filepath = os.path.join('results2/', alg_name) + '.p'
    
    pickle.dump(stats, open(data_filepath,'wb'))
    plot_reward_curve(stats, alg_name)

def train(alg_type):
    TRIALS = 5
    ITERATIONS = 4
    SAMPLES_PER_ROLLOUT = 20 # keep as multiple of 10
    SAMPLES_PER_EVAL = 20

    # TRIALS = 1
    # ITERATIONS = 2
    # SAMPLES_PER_ROLLOUT = 10
    # SAMPLES_PER_EVAL = 2


    overall_stats = []

    print("DEBUG!!")
    weight_id = None
    params = None

    env = ray.env.env
    supervisor = ray.env.supervisor
    print "Agent..."
    main_agent = RayDartAgent(DeepLearner(gpu=True), 
                                env, 
                                supervisor)
    main_agent.setup()


    # if(alg_type == 'dagger'):

    #     main_agent = ray.env.dagger_agent
    # elif(alg_type == 'off_d'):
    #     main_agent = ray.env.off_agent
    # elif(alg_type == 'dart'):
    #     main_agent = ray.env.dart_agent

    for i in range(TRIALS):
        trial_stats = []
        eps = 0.5
        for j in range(ITERATIONS):
            # print("Agent {}: Trial {}, Iteration {}".format(agent_name, i, j))
            weights = main_agent.get_weights()
            weight_id = ray.put(weights)
            
            if(alg_type == 'dagger'):
                params = [j]
            elif(alg_type == 'dart_off'):
                params = [0.0]#[main_agent.compute_eps()]
            elif(alg_type == 'off_d'):
                params = [1]
            rollouts = [rollout.remote(weight_id, params,alg_type, c=k) for k in range(SAMPLES_PER_ROLLOUT)]

            results = []
            # for i in range(10):
            #     batch_res, rollouts = ray.wait(rollouts, num_returns=SAMPLES_PER_ROLLOUT / 10)
            #     batch_res_vals = ray.get(batch_res)
            #     new_batch_res_vals = []
            #     for x, y in batch_res_vals:
            #         new_x = np.empty_like(x)
            #         new_x[:] = x[:]
            #         new_y = np.empty_like(y)
            #         new_y[:] = y[:]
            #         new_batch_res_vals.append((new_x, new_y))
            #     batch_res_vals = new_batch_res_vals

            #     results.extend(batch_res_vals)
            #     print "Collected batch of %d" % len(batch_res)

            t_results = ray.get(rollouts)
            t_results = [x[0] for x in t_results]
            state_list = [np.array(res[0], copy=True) for res in t_results]
            action_list = [np.array(res[1], copy=True) for res in t_results]
            t_results = []

            print "Collected Rollouts"
            # state_list, action_list = zip(*results)
            #state_list = [np.copy(x) for x in state_list]

           
            main_agent.update_model(state_list, action_list)

           

            print "EVAL MODEL"
            for k in range(SAMPLES_PER_EVAL):
                main_agent.eval_policy()
            trial_stats.append(main_agent.get_statistics())
            print "GET STATISTICS MAY NOT BE COMPLETELY CORRECT"

            if(alg_type == 'dart'):
                eps = main_agent.cal_eps()
                print "CHOSEN EPS ",eps


        overall_stats.append(trial_stats)
        print("Stats for Trial {}: ".format(i))
        print("Average train acc, test acc, reward, surrogate loss")
        print(overall_stats[i])
        main_agent.reset()
    save_data(overall_stats,alg_type)

if __name__ == '__main__':
    # import sys
    # AGENT = sys.argv[1]

    #train('off_d')

    #train('dagger')

    train('dart_off')
