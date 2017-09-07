import numpy as np
from numpy.random import rand,randint
from gym_driving.agents.supervised_agent import *
import IPython

class DartAgent(SupervisedAgent):
    def __init__(self, learner, env, supervisor):
        super(DartAgent, self).__init__(learner, env, supervisor)
        self.eps = 0.1

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
            sample = rand()

            if sample < self.eps:
                action = [randint(5), 1]
            else:
                action = [supervisor_label, 1]

            surrogate_losses.append(0)
            next_state, reward, done, info_dict = self.env._step(action)
            reward = 0

            states.append(state)
            actions.append(supervisor_label)
            rewards.append(reward)
            supervisor_labels.append(supervisor_label)

            state = next_state

        self.rewards.append(sum(rewards))
        self.surrogate_losses.append(sum(surrogate_losses))

        return states, actions


    def cal_eps(self):

        self.rewards
        num_samples = len(self.surrogate_losses)

        cumm_loss = 0.0
        for i in range(num_samples):

            r = float(self.rewards[i])
            l = float(self.surrogate_losses[i])

            scaled_l = 100.0*l/r
            cumm_loss += scaled_l

        cum_loss = cumm_loss/float(num_samples)
        print "CUM LOSS ",cum_loss
        e_range = np.linspace(0,1.0,num=100)
        sol = []
        for e in e_range:
            val = (e/5.0)**(cum_loss)*(1-e)**(100.0-cum_loss)
            sol.append(val)
                #IPython.embed()
        idx = np.argmax(sol)

        return e_range[idx]
