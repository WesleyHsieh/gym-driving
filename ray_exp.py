import gym
import numpy as np
import tensorflow as tf
import time
import ray
from ray_dagger_agent import RayDaggerAgent
# from deep_lfd.learning_driving.linear_learner import *
from deep_lfd.learning_driving.deep_learner import *
from gym_driving.envs.agents.supervised_agent import *
from gym_driving.envs.agents.dagger_agent import *
from gym_driving.envs.agents.driving_agent import *
from gym_driving.envs.driving_env import *

agent_name = "test"
NUM_WORKERS = 2
os.environ["SDL_VIDEODRIVER"] = "dummy"

ray.init(num_workers=NUM_WORKERS)

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

def agent_init():
    env = ray.env.env
    supervisor = ray.env.supervisor
    return RayDaggerAgent(DeepLearner(), env, supervisor)

def agent_reinit(agent):
    agent.reset()
    return agent

ray.env.agent = ray.EnvironmentVariable(agent_init, agent_reinit)

@ray.remote
def rollout(weights):
    agent = ray.env.agent
    agent.set_weights(weights)
    return agent.rollout_algorithm()


def save_data(self, stats, file_path="tmp/"):
    data_filepath = os.path.join(file_path, agent_name) + '.pkl'
    pickle.dump(stats, open(data_filepath,'wb'))
    self.plot_reward_curve(stats, agent_name)

def train():
    TRIALS = 5
    ITERATIONS = 5
    SAMPLES_PER_ROLLOUT = 1
    SAMPLES_PER_EVAL = 5
    overall_stats = [] 
    main_agent = ray.env.agent

    for i in range(TRIALS):
        trial_stats = []

        for j in range(ITERATIONS):
            print("Agent {}: Trial {}, Iteration {}".format(agent_name, i, j))
            weights = main_agent.get_weights()
            weight_id = ray.put(weights)

            rollouts = [rollout.remote(weight_id) for k in range(SAMPLES_PER_ROLLOUT)]
            results = ray.get(rollouts)


            main_agent.update_model(state_list, action_list)
            for k in range(SAMPLES_PER_EVAL):
                main_agent.eval_policy()
            trial_stats.append(main_agent.get_statistics())
            print "GET STATISTICS MAY NOT BE COMPLETELY CORRECT"
        overall_stats.append(trial_stats)
        print("Stats for Trial {}: ".format(i))
        print("Average train acc, test acc, reward, surrogate loss")
        print(overall_stats[i])
        agent.reset()
    save_data(overall_stats)

if __name__ == '__main__':
    train()
