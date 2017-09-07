#from svm import LinearSVM
#from net import Net
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import IPython
import cPickle

from state import State
class Analysis():

    def compute_std_er_m(self,data):
        n = data.shape[0]
        std = np.std(data)

        return std/np.sqrt(n)

    def compute_m(self,data):
        n = data.shape[0]
        return np.sum(data)/n

    def get_perf(self,data, color=None):
        #SAve each mean and err at the end
        iters = data.shape[1]
        mean = np.zeros(iters)
        err = np.zeros(iters)
        x = np.zeros(iters)

        for i in range(iters):
            mean[i] = self.compute_m(data[:,i]) # for iteration i, compute mean reward across trials
            x[i] = i
            err[i] = self.compute_std_er_m(data[:,i])
        if color is None:
            plt.errorbar(x,mean,yerr=err,linewidth=5.0)
        else:
            plt.errorbar(x,mean,yerr=err,linewidth=5.0, color=color)


        self.mean = mean
        self.err = err
        self.x = x

        return [mean,err]





    @staticmethod
    def load(filename):
        a = cPickle.load(open(filename, 'rb'))
        if a.x is not None and a.mean is not None:
            a.set_errorbar()
        return a



    def plot(self, names = None, label = None, filename=None, ylims=None):
        if label is None:
            label = 'Reward'
        plt.ylabel(label)
        plt.xlabel('Iterations')

        if names is None:
            names = ['Sup']
            #names = ['NN_Supervise','LOG_Supervisor']
        plt.legend(names,loc='upper center',prop={'size':10}, bbox_to_anchor=(.5, 1.12), fancybox=False, ncol=len(names))

        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22},

        axes = plt.gca()
        axes.set_xlim([0,self.iters])
        if not ylims is None:
            axes.set_ylim(ylims)
            #axes.set_ylim([-60, 100])
        if filename is not None:
            plt.savefig(filename, format='eps', dpi=1000)

        plt.ioff()
        plt.clf()
        plt.cla()
        plt.close()
        #plt.show(block=False)
        #plt.close()









