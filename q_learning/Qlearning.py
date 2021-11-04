#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import numpy as np
import matplotlib.pyplot as plt
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from env_custom import *
import gym
import pandas as pd
#pd.options.plotting.backend = "plotly"
import itertools as it

import plotly.express as px
global colors
import json
from datetime import datetime
import math
import pickle
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
#%%



def maxAction(Q, state):
    actionValue = Q[state].max()
    action = np.argmax(Q[state])
    return action, actionValue


def create_bins(x_threshold,theta_dot_threshold, nBins):
    ## 5 observations, x, x_dot, cos(theta), sin(theta), theta_dot
    # bins = np.zeros((5, nBins))
    bins1 = np.linspace(-x_threshold, x_threshold, nBins[0])
    bins2 = np.linspace(-1, 1, nBins[1])
    bins3 = np.linspace(-1, 1, nBins[2])
    bins4 = np.linspace(-1, 1, nBins[3])
    bins5 = np.linspace(-theta_dot_threshold, theta_dot_threshold, nBins[4])
    bins = [bins1, bins2, bins3, bins4, bins5]
    return bins


def assignBins(observation, bins, observationNum):
    state = np.zeros(observationNum)
    for i in range(observationNum):
        state[i] = np.digitize(observation[i], bins[i])
    tmp = [int(x) for x in state]
    state = tuple(tmp)
    return state

# def get_epsilon(t, min_epsilon, STEPS_TO_TRAIN):
#     # eps = max(min_epsilon, min(1, 1 - math.log10((t + 1) / decay)))
#     eps = max(min_epsilon, 1 - 2*t/(STEPS_TO_TRAIN))
#     return eps

def get_epsilon(t, min_epsilon, decay):
    return max(min_epsilon, min(1., 1. - math.log10((t + 1) / decay)))

# def get_learning_rate(t, min_lr, decay):
#     return max(min_lr, min(ALPHA0, 1. - math.log10((t + 1) / decay)))

def choose_action(Q, state, EPS):
    if (np.random.random() < EPS):
        return env.action_space.sample()
    else:
        action, actionValue = maxAction(Q, state)
        return action


def initialize_Q(observationNum, actionNum, nBins):
    columns = list(it.product(range(nBins[0]+1), range(nBins[1]+1), range(nBins[2]+1), range(nBins[3]+1), range(nBins[4]+1)))
    index = range(actionNum)
    Q = pd.DataFrame(0.00000000, index=index, columns=columns)
    return Q

def play_one_episode(bins, Q, EPS, ALPHA, observationNum, render=False):
    observation = env.reset()
    # if render:
    #     env.render()
    done = False
    cnt = 0  # number of moves in an episode
    state = assignBins(observation, bins, observationNum)
    # print('initial state', state)
    totalReward = 0
    act = choose_action(Q, state, EPS)
    # print('initial action', act)
    dList = [{'timestep':0, 'x':observation[0], 'x_dot':observation[1], 'costheta':observation[2], 'sintheta':observation[3], 'theta_dot':observation[4], 'action':0, 'reward':0}]

    while not done:
        cnt += 1
        observationNew, reward, done, _ = env.step(act)
        stateNew = assignBins(observationNew, bins, observationNum)
        totalReward += reward
        actionNew = choose_action(Q, stateNew, EPS)
        a_, target_values = maxAction(Q, stateNew)
        update = ALPHA * (reward + GAMMA * target_values - Q.at[act, state])
        # print('update', update)
        # print('state', state)
        # print('act', act)
        Q.at[act, state] += update
        # print('Q.at[act, state]',Q.at[act, state])
        state, act = stateNew, actionNew
        # print('new state', stateNew, 'new action', actionNew, 'reward', reward)
        # print('Qmax', Q.max().max())
        # print('Qmin', Q.min().min())
        if render:
            dList.append({'timestep':cnt, 'x':observationNew[0], 'x_dot':observationNew[1], 'costheta':observationNew[2], 'sintheta':observationNew[3], 'theta_dot':observationNew[4], 'action':actionNew, 'reward':reward})
            #env.render()
    return totalReward, cnt, state, act, Q, dList

def play_many_episodes(observationNum, actionNum, nBins, numEpisode, min_epsilon, min_lr):

    Q = initialize_Q(observationNum, actionNum, nBins)
    length = []
    reward = []
    eps = []
    for n in range(numEpisode+1):
        # eps=0.5/(1+n*10e-3)
        EPS = get_epsilon(n, min_epsilon, decay)
        # ALPHA = 0.1get_learning_rate(n, min_lr, decay)
        # ALPHA = ALPHA0
        ALPHA = 0.1
        # episodeReward, episodeLength, state, act, Q = play_one_episode(bins, Q, EPS, ALPHA, observationNum)

        if n % 10000 == 0:
            # print(n, '%.4f' % EPS, episodeReward)
            episodeReward, episodeLength, state, act, Q, dList = play_one_episode(bins, Q, EPS, ALPHA, observationNum, render=True)
            df = pd.DataFrame.from_dict(dList)
            df.to_csv(str(n)+'_ep.csv')
            print('{}, \t {:.4f}, \t {}, \t {}, \t {}'.format(n, EPS, episodeReward, state, episodeLength))
        else:
            episodeReward, episodeLength, state, act, Q, dList = play_one_episode(bins, Q, EPS, ALPHA, observationNum, render=False)

            # print('Qmax', Q.max().max())
        if n % 50000 ==0:
            Q.T.to_csv('Q_'+str(n)+'.csv')

        length.append(episodeLength)
        reward.append(episodeReward)
        eps.append(EPS)

    result = pd.DataFrame({'reward':reward, 'episodeLength':length, 'eps':eps, 'lastState': str(state)})

    return result, Q

#%%
if __name__ == '__main__':


    numEpisode=100000
    EP_STEPS=800
    Te=0.05
    resetMode='experimental'

    # env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode=resetMode, sparseReward=False,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)#,integrator='ode')#,integrator='rk4')
    env = CartPoleRK4(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode=resetMode, sparseReward=False,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)#,integrator='ode')#,integrator='rk4')

    actionNum = env.action_space.n
    observationNum = env.observation_space.shape[0]

    ALPHA0 = 1
    GAMMA = 0.99
    decay = numEpisode/10
    min_epsilon = 0.1
    min_lr = 0.1

    x_threshold = env.x_threshold
    theta_dot_threshold = 12
    nBins = [10, 10, 10, 10, 10]
    # nBins = [30, 30, 50, 50, 50]
    INFO = {'ALPHA0': ALPHA0, 'GAMMA': GAMMA, 'decay':decay, 'min_epsilon':min_epsilon, 'min_lr':min_lr, 'numEpisode': numEpisode, 'resetMode':resetMode, 'theta_dot_threshold':theta_dot_threshold, 'nBins':str(nBins), 'reward':'with limit theta dot'}



    import time
    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")
    logpath = './results/'+dt_string
    os.makedirs(logpath, exist_ok=True)
    os.chdir(logpath)
    with open('0_INFO.json', 'w') as file:
        json.dump(INFO, file, indent=4)

    bins = create_bins(x_threshold, theta_dot_threshold, nBins=nBins)
    result, Q = play_many_episodes(observationNum, actionNum, nBins, numEpisode, min_epsilon, min_lr)
    print('finished')
    
    result.to_csv('ql_alpha0_'+str(ALPHA0)+'_gamma_'+str(GAMMA)+'_minEps_'+str(min_epsilon)+'.csv')
    Q.T.to_csv('Q_alpha0_'+str(ALPHA0)+'_gamma_'+str(GAMMA)+'_minEps_'+str(min_epsilon)+'.csv')


# %%
