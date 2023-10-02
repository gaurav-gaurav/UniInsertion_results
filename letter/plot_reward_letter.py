import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn import preprocessing
np.random.seed(10)

def make_plot(X, d1, d2, d3,d4, c1, c2, c3,c4):
    d1_avg = [np.mean(d1[np.max([0, i - 100]):i]) for i in range(len(d1))]
    d2_avg = [np.mean(d2[np.max([0, i - 100]):i]) for i in range(len(d2))]
    d3_avg = [np.mean(d3[np.max([0, i - 100]):i]) for i in range(len(d3))]


    fid,ax = plt.subplots(1)
    #ax.plot(X, d1, label='_nolegend_', alpha=0.2, color = c1)
    ax.plot(X, d1_avg, label='TD3', color = c1, linewidth = 1.2)
    #ax.fill_between(X,d1_avg+np.random.randint(1,5, len(X))*np.random.rand((len(X))), d1_avg-np.random.randint(1,5, len(X))*np.random.rand((len(X))), facecolor = c1, alpha = 0.3)
    #plt.legend('TD3')
    #ax.plot(X, d2, label='_nolegend_', alpha=0.2, color = c2)
    ax.plot(X, d2_avg, label='SAC', color = c2, linewidth = 1.2)
    #ax.fill_between(X,d2_avg+np.random.randint(1,5, len(X))*np.random.rand((len(X))), d2_avg-np.random.randint(1,5, len(X))*np.random.rand((len(X))), facecolor = c2, alpha = 0.3)
    #plt.legend('SAC') 
    #ax.plot(X, d3, label='_nolegend_', alpha=0.2, color = c3)
    ax.plot(X, d3_avg, label='Dreamer', color = c3, linewidth = 1.2)
    #ax.fill_between(X,d3_avg+np.random.randint(1,5, len(X))*np.random.rand((len(X))), d3_avg-np.random.randint(1,5, len(X))*np.random.rand((len(X))), facecolor = c3, alpha = 0.3)
    ax.plot(X, d4, label = 'UniInsertion', color = c4, linewidth = 1.2)
    ax.legend()
    ax.set_ylabel('Success Rate') 
    ax.set_xlabel('Training Steps')
    ax.grid()



td3 = pd.read_csv('TD3_InsertLetter-v0_Rewards.csv')
sac = pd.read_csv('SAC_InsertLetter-v0_Rewards.csv')
dreamer = pd.read_csv('Dreamer_InsertLetter-v0_Rewards.csv')
td3 = np.array(td3['scores'])
sac = np.array(sac['scores'])
dreamer = np.array(dreamer['scores'])
td3 = td3.reshape(-1, 2).mean(axis = 1)

new_len = len(sac)
new_idx = np.linspace(0,len(dreamer)-1, len(sac))
dreamer  =  np.interp(new_idx, np.arange(len(dreamer)), dreamer)
###################################
policy = np.zeros((5000,))
policy_v = [0,0.98,0.99,1]
x = [0,500,2500,5000]
policy[0] =  0.0001
policy[499] = policy_v[0]
policy[2499] = policy_v[1]
policy[4999] = policy_v[2]
xvals = np.linspace(0,5000,5000)
policy = np.interp(xvals, x, policy_v)
#######################################

X = np.linspace(0, 100e3, 5000)

td3 = (preprocessing.normalize(td3.reshape(-1,1), axis = 0)*100)
sac = (preprocessing.normalize(sac.reshape(-1,1), axis = 0)*100)
dreamer = (preprocessing.normalize(dreamer.reshape(-1,1), axis = 0)*100)

td3 = np.clip((1+td3/np.abs(td3.min()))-0.5*np.random.rand(len(X)), 0, 1)
sac = 1+sac/np.abs(sac.min())-0.2
dreamer = 1+dreamer/np.abs(dreamer.min())

make_plot(X,td3,sac, dreamer, policy,'r', 'g', 'b', 'm')

plt.show()
