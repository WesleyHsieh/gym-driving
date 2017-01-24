from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

import gym_driving


# from gym.envs.registration import spec

# print('Environment Spec: ', spec('DrivingEnv-v0'))

stub(globals())

# env = GymEnv('DrivingEnv-v0')
env = normalize(GymEnv('Driving_Gym-v0'))

# policy = CategoricalMLPPolicy(
#     env_spec=env.spec,
#     # The neural network policy should have two hidden layers, each with 32 hidden units.
#     hidden_sizes=(256, 256, 256)
# )

policy = CategoricalGRUPolicy(
    env_spec=env.spec,
    hidden_dim=32,
)


baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=env.horizon,
    n_itr=1000,
    discount=0.99,
    step_size=0.0001,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

# algo = TRPO(
#     env=env,
#     policy=policy,
#     baseline=baseline,
#     batch_size=4000,
#     max_path_length=env.horizon,
#     n_itr=50,
#     discount=0.99,
#     step_size=0.01,
#     optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
# )

# algo.train()

run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=-1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
