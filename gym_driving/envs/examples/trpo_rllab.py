from rllab.algos.vpg import VPG
from rllab.algos.tnpg import TNPG
from rllab.algos.erwr import ERWR
from rllab.algos.reps import REPS
from rllab.algos.trpo import TRPO
from rllab.algos.cem import CEM
from rllab.algos.cma_es import CMAES
from rllab.algos.ddpg import DDPG

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import gym_driving

import IPython

def run_task(*_):
    import gym_driving
    env = normalize(GymEnv('DrivingEnv-v0'))
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=40000,
        max_path_length=env.horizon,
        n_itr=250,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=6,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    exp_name='trpo',
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)