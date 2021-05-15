#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:27:33 2021

@author: lfu
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from env_custom import CartPoleButter
import gym
import pandas as pd
import itertools as it

import plotly.express as px
from bokeh.plotting import figure, output_file, show, save
from bokeh.palettes import d3, viridis
from bokeh.models import Range1d, Legend, Span
global colors
import json
colors = d3["Category20"][20]
from datetime import datetime
import math
import pickle
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback

def plot_df(df, x_col, y_list, title='title', symbol='line', index=False, shiftX=0, shiftY=0):
    fig = figure(
    tools = "pan,wheel_zoom,box_zoom,box_select,reset,crosshair,hover,undo,redo,save",
    active_drag="box_zoom",
    active_scroll="wheel_zoom",
    title = title,
    plot_width=1200, plot_height=600
    )
    fname = title+'.html'
    output_file(fname, mode='inline')

    if type(y_list) != list:
        y_list = [y_list]

    if index:
        xplot = df.index.values
    elif x_col == 'index':
        xplot = df.index.values
    else:
        xplot = df[x_col]
    xplot += shiftX

    legend_it = []
    if symbol == 'line':
        i=0
        for y_col in y_list:
            color = colors[i % len(colors)]
            yplot = df[y_col] + shiftY
            c = fig.line(xplot, yplot, line_width=2, alpha=0.8, color = color)
            legend_it.append((str(y_col), [c]))
            i += 1
    elif symbol == 'circle':
        i=0
        for y_col in y_list:
            color = colors[i % len(colors)]
            yplot = df[y_col] + shiftY
            c = fig.circle(xplot, yplot, fill_alpha=0.6, size=10, color = color)
            legend_it.append((str(y_col), [c]))
            i += 1
    elif symbol == 'line+circle':
        i=0
        for y_col in y_list:
            color = colors[i % len(colors)]
            yplot = df[y_col] + shiftY
            c1 = fig.circle(xplot, yplot, fill_alpha=0.6, size=10, color = color)
            c2 = fig.line(xplot, yplot, line_width=2, alpha=0.8, color = color)
            legend_it.append((str(y_col), [c]))
            i += 1
    legend = Legend(items=legend_it, location=(10, 20))
    legend.click_policy="hide"
    fig.xaxis.axis_label = str(x_col)
    fig.yaxis.axis_label = str(y_col)
#    fig.legend.location = "bottom_left"
#    fig.legend.click_policy="hide"
    fig.add_layout(legend, 'right')
    save(fig)
    return fig


def maxAction(Q, state):
    actionValue = Q[state].max()
    action = np.argmax(Q[state])
    return action, actionValue


def create_bins(x_threshold,theta_dot_threshold, nBins):
    ## 5 observations, x, x_dot, cos(theta), sin(theta), theta_dot
    bins = np.zeros((5, nBins))
    bins[0] = np.linspace(-x_threshold, x_threshold, nBins)
    bins[1] = np.linspace(-5, 5, nBins)
    bins[2] = np.linspace(-1, 1, nBins)
    bins[3] = np.linspace(-1, 1, nBins)
    bins[4] = np.linspace(-theta_dot_threshold, theta_dot_threshold, nBins)
    return bins


def assignBins(observation, bins, observationNum):
    state = np.zeros(observationNum)
    for i in range(observationNum):
        state[i] = np.digitize(observation[i], bins[i])
    tmp = [int(x) for x in state]
    state = tuple(tmp)
    return state

def get_epsilon(t, min_epsilon, STEPS_TO_TRAIN):
    # return max(min_epsilon, min(1, 1 - math.log10((t + 1) / decay)))
    eps = max(min_epsilon, 1 - 2*t/(STEPS_TO_TRAIN))
    return eps

def choose_action(Q, state, EPS):
    if (np.random.random() < EPS):
        return env.action_space.sample()
    else:
        action, actionValue = maxAction(Q, state)
        return action


def initialize_Q(observationNum, actionNum, nBins):
    columns = list(it.product(range(nBins+1), repeat=observationNum))
    index = range(actionNum)
    Q = pd.DataFrame(0, index=index, columns=columns)
    return Q

def play_one_episode(bins, Q, EPS, observationNum):
    observation = env.reset()
    done = False
    cnt = 0  # number of moves in an episode
    state = assignBins(observation, bins, observationNum)
    # print('initial state', state)
    totalReward = 0
    act = choose_action(Q, state, EPS)
    # print('initial action', act)

    while not done:
        cnt += 1
        observationNew, reward, done, _ = env.step(act)
        stateNew = assignBins(observationNew, bins, observationNum)
        totalReward += reward
        actionNew = choose_action(Q, stateNew, EPS)
        
        a_, target_values = maxAction(Q, stateNew)
        Q[state][act] += ALPHA * (reward + GAMMA * target_values - Q[state][act])
        state, act = stateNew, actionNew
        # print('new state', stateNew, 'new action', actionNew, 'reward', reward)



    return totalReward, cnt, state, act

def play_many_episodes(observationNum, actionNum, nBins, numEpisode, min_epsilon):
    global Q
    Q = initialize_Q(observationNum, actionNum, nBins)

    length = []
    reward = []
    eps = []
    for n in range(numEpisode):
        # eps=0.5/(1+n*10e-3)
        EPS = get_epsilon(n, min_epsilon, numEpisode)

        episodeReward, episodeLength, state, act = play_one_episode(bins, Q, EPS, observationNum)

        if n % 1000 == 0:
            # print(n, '%.4f' % EPS, episodeReward)
            print('{}, \t {:.4f}, \t {}, \t {}'.format(n, EPS, episodeReward, state))
            print(Q[state])
        length.append(episodeLength)
        reward.append(episodeReward)
        eps.append(EPS)

    result = pd.DataFrame({'reward':reward, 'episodeLength':length, 'eps':eps})
    return result

#%%
if __name__ == '__main__':


    numEpisode=1e6
    EP_STEPS=800
    Te=0.05

    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode='experimental', sparseReward=False,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)#,integrator='ode')#,integrator='rk4')


    actionNum = env.action_space.n
    observationNum = env.observation_space.shape[0]

    ALPHA = 0.1
    GAMMA = 0.95
    decay = 10000
    min_epsilon = 0.01

    x_threshold = env.x_threshold
    theta_dot_threshold = 2*np.pi
    nBins = 15

    import time
    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")

    bins = create_bins(x_threshold, theta_dot_threshold, nBins=nBins)
    result = play_many_episodes(observationNum, actionNum, nBins, numEpisode, min_epsilon)
    print('finished')
    
    result.to_csv('ql_alpha_'+str(ALPHA)+'_gamma_'+str(GAMMA)+'_'+dt_string+'_nBins_'+str(nBins)+'.csv')

    elapsed_time = time.time() - start_time
    print('Elapsed time', elapsed_time)




