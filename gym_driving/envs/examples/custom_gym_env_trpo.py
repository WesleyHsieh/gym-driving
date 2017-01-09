from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

import gym_driving


# from gym.envs.registration import spec

# print('Environment Spec: ', spec('DrivingEnv-v0'))

# stub(globals())

env = GymEnv('DrivingEnv-v0')

policy = CategoricalMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)


baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=1000,
    max_path_length=env.horizon,
    n_itr=10,
    discount=0.99,
    step_size=0.01,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

algo.train()

# run_experiment_lite(
#     algo.train(),
#     # Number of parallel workers for sampling
#     n_parallel=-1,
#     # Only keep the snapshot parameters for the last iteration
#     snapshot_mode="last",
#     # Specifies the seed for the experiment. If this is not provided, a random seed
#     # will be used
#     seed=1,
#     # plot=True,
# )
