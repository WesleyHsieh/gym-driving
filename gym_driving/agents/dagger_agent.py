import numpy as np
from gym_driving.agents.supervised_agent import *
import IPython

class DaggerAgent(SupervisedAgent):
    def __init__(self, learner, env, supervisor):
        super(DaggerAgent, self).__init__(learner, env, supervisor)

    def rollout_algorithm(self):
        """
        Rollout algorithm on underlying environment
        for one trajectory, using supervisor.
        """
        states, actions, rewards, supervisor_labels, surrogate_losses = [], [], [], [], []
        done = False
        state = self.env._reset()
        while not done:
            supervisor_label = self.supervisor.eval_policy(self.env)
            if self.iterations > 0:
                action = self.learner.eval_policy(state)
                surrogate_losses.append(action != supervisor_label)
                action_packed = [action, 1]
                next_state, reward, done, info_dict = self.env._step(action_packed)
            else:
                action = supervisor_label
                surrogate_losses.append(0)
                action_packed = [action, 1]
                next_state, reward, done, info_dict = self.env._step(action_packed)
                reward = 0

            states.append(state)
            actions.append(supervisor_label)
            rewards.append(reward)
            supervisor_labels.append(supervisor_label)

            state = next_state

        self.rewards.append(sum(rewards))
        self.surrogate_losses.append(sum(surrogate_losses))

        return states, actions
