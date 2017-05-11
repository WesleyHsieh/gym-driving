import gym
import numpy as np
import time
import ray
import IPython
import os
import argparse
import pickle

from gym_driving.envs.driving_env import *


os.environ["SDL_VIDEODRIVER"] = "dummy"
config_filepath = "gym_driving/envs/configs/driving_experiment_config.json"
def ray_exp(num_workers):
	def env_init():
		return DrivingEnv(render_mode=False, config_filepath=config_filepath)

	def env_reinit(env):
		return env

	@ray.remote
	def rollout(iters):
		env = ray.env.env
		count = 0
		for _ in range(iters):
			env.step(0)
			count += 1
		return count
	start = time.time()
	iters = 10000 / num_workers
	ray.env.env = ray.EnvironmentVariable(env_init, env_reinit)
	rollouts = [rollout.remote(iters) for k in range(num_workers)]
	results = ray.get(rollouts)
	end = time.time()
	print(results)
	print("Time Elapsed", end - start)
	return end - start

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_workers", help="num_workers", default=1)
	args = parser.parse_args()

	num_workers = int(args.num_workers)
	ray.init(num_workers=num_workers)
	num_experiments = 100
	results = [ray_exp(num_workers) for _ in range(num_experiments)]

	if num_workers == 1:
		myfile = open('out.pkl', 'w')
		result_dict = {}
		result_dict[num_workers] = results
		pickle.dump(result_dict, myfile)
		myfile.truncate()
	else:
		myfile = open('out.pkl', 'r+')
		result_dict = pickle.load(myfile)
		myfile.seek(0)
		result_dict[num_workers] = results
		pickle.dump(result_dict, myfile)
		myfile.truncate()