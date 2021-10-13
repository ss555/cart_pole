#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:50:16 2021

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
pd.options.plotting.backend = "plotly"
import plotly.express as px
from bokeh.plotting import figure, output_file, show, save
from bokeh.palettes import d3, viridis
from bokeh.models import Range1d, Legend, Span
global colors
import json
colors = d3["Category20"][20]
from datetime import datetime
import math
import random
import itertools as it
#%%
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
#    fig.output_backend = "svg"

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
    # bins = np.zeros((5, nBins))
    bins1 = np.linspace(-x_threshold, x_threshold, nBins[0])
    bins2 = np.linspace(-5, 5, nBins[1])
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


def get_epsilon(t, min_epsilon, decay):
        return max(min_epsilon, min(1., 1. - math.log10((t + 1) / decay)))

def get_learning_rate(t, min_lr, decay):
    return max(min_lr, min(ALPHA0, 1. - math.log10((t + 1) / decay)))

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

def update_model(model, past_state, past_action, state, reward):
        """updates the model

        Args:
            past_state       (int): s
            past_action      (int): a
            state            (int): s'
            reward           (int): r
        Returns:
            Nothing
        """
        # Update the model with the (s,a,s',r) tuple (1~4 lines)
        model.update({(past_state, past_action): (state, reward)})
        return model

def planning_step(past_state, past_action, planning_steps, Q, model):
    """performs planning, i.e. indirect RL.

    """
    # The indirect RL step:
    # - Choose a state and action from the set of experiences that are stored in the model. 
    # - Query the model with this state-action pair for the predicted next state and reward.(~1 line)
    # - Update the action values with this simulated experience. 
    # - Repeat for the required number of planning steps.
    #
    # Note that the update equation is different for terminal and non-terminal transitions.
    # To differentiate between a terminal and a non-terminal next state, assume that the model stores
    # the terminal state as a dummy state like -1
    #
    # Important: remember you have a random number generator 'planning_rand_generator' as
    #     a part of the class which you need to use as self.planning_rand_generator.choice()
    #     For the sake of reproducibility and grading, *do not* use anything else like
    #     np.random.choice() for performing search control.

    for i in range(planning_steps):
        t = list(model.keys())
        tr = tuple(random.sample(t, len(t))) ## shuffle the list
        next_state, reward = model[tr[0]]

        q_max = Q[next_state].max()

        # Q[s,a] = Q[s,a] + ALPHA*(reward + GAMMA*target_values - Q[s,a])
        update = ALPHA * (reward + GAMMA * q_max - Q.at[past_action, past_state])
        Q.at[past_action, past_state] += update 
    return Q

def play_one_episode(bins, Q, EPS, ALPHA, observationNum, model, render=False):

    observation = env.reset()
    if render:
        env.render()
    done = False
    cnt = 0  # number of moves in an episode
    state = assignBins(observation, bins, observationNum)
    # print('initial state', state)
    totalReward = 0
    act = choose_action(Q, state, EPS)
    # print('initial action', act)
    dList = [{'timestep':0, 'x':observation[0], 'x_dot':observation[1], 'costheta':observation[2], 'sintheta':observation[3], 'theta_dot':observation[4], 'action':0}]

    while not done:
        cnt += 1
        observationNew, reward, done, _ = env.step(act)
        stateNew = assignBins(observationNew, bins, observationNum)
        totalReward += reward
        a_, target_values = maxAction(Q, stateNew)
        update = ALPHA * (reward + GAMMA * target_values - Q.at[act, state])
        # print('update', update)
        # print('state', state)
        # print('act', act)
        Q.at[act, state] += update

        # model = update_model(model, state,act,stateNew,reward)
        # Q = planning_step(state, act, planning_steps, Q, model)
        model.update({(state, act): (stateNew, reward)})
        for i in range(planning_steps):
            t = list(model.keys())
            tr = tuple(random.sample(t, len(t))) ## shuffle the list
            next_state, reward = model[tr[0]]
            q_max = Q[next_state].max()
            # Q[s,a] = Q[s,a] + ALPHA*(reward + GAMMA*target_values - Q[s,a])
            update_planning = ALPHA * (reward + GAMMA * q_max - Q.at[act, state])
            # print('update_planning', update_planning)
            Q.at[act, state] += update_planning


        actionNew = choose_action(Q, stateNew, EPS)
        
        # print('Q.at[act, state]',Q.at[act, state])
        state, act = stateNew, actionNew
        # print('new state', stateNew, 'new action', actionNew, 'reward', reward)
        # print('Qmax', Q.max().max())
        # print('Qmin', Q.min().min())
        # env.render()
        if render:
            dList.append({'timestep':cnt, 'x':observationNew[0], 'x_dot':observationNew[1], 'costheta':observationNew[2], 'sintheta':observationNew[3], 'theta_dot':observationNew[4], 'action':actionNew})
            env.render()
    return totalReward, cnt, state, act, Q, dList, model

def play_many_episodes(observationNum, actionNum, nBins, numEpisode, min_epsilon, min_lr, model):

    Q = initialize_Q(observationNum, actionNum, nBins)

    length = []
    reward = []
    eps = []
    for n in range(numEpisode+1):
        # eps=0.5/(1+n*10e-3)
        EPS = get_epsilon(n, min_epsilon, decay)
        ALPHA = get_learning_rate(n, min_lr, decay)
        # ALPHA = ALPHA0

        

        if n % 1000 == 0:
            # print(n, '%.4f' % EPS, episodeReward)
            episodeReward, episodeLength, state, act, Q, dList, model = play_one_episode(bins, Q, EPS, ALPHA, observationNum, model, render=False)
            df = pd.DataFrame.from_dict(dList)
            df.to_csv(str(n)+'_ep.csv')
            print('{}, \t {:.4f}, \t {}, \t {}, \t {}'.format(n, EPS, episodeReward, state, episodeLength))
        else:
            episodeReward, episodeLength, state, act, Q, dList, model = play_one_episode(bins, Q, EPS, ALPHA, observationNum, model, render=False)
            # print('{}, \t {:.4f}, \t {}, \t {}, \t {}'.format(n, EPS, episodeReward, state, episodeLength))

            # print('Qmax', Q.max().max())
        if n % 50000 ==0:
            Q.T.to_csv('Q_'+str(n)+'.csv')

        length.append(episodeLength)
        reward.append(episodeReward)
        eps.append(EPS)

    result = pd.DataFrame({'reward':reward, 'episodeLength':length, 'eps':eps, 'lastState': str(state)})

    return result, Q, model


#%%
if __name__ == '__main__':
    numEpisode=100000
    EP_STEPS=800
    Te=0.05
    resetMode='experimental'
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode=resetMode, sparseReward=False,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)#,integrator='ode')#,integrator='rk4')
    env = env.unwrapped

    actionNum = env.action_space.n
    observationNum = env.observation_space.shape[0]

    ALPHA0 = 0.1
    GAMMA = 0.99
    numGames = 200
    decay = 1000
    min_epsilon = 0.01
    planning_steps = 1
    min_lr = 0.1

    rand_generator = np.random.RandomState(42)
    planning_rand_generator = np.random.RandomState(42)

    x_threshold = env.x_threshold
    theta_dot_threshold = 2*np.pi
    nBins = [10, 10, 10, 10, 10]
    INFO = {'ALPHA0': ALPHA0, 'GAMMA': GAMMA, 'decay':decay, 'min_epsilon':min_epsilon, 'min_lr':min_lr, 'numEpisode': numEpisode, 'resetMode':resetMode, 'theta_dot_threshold':theta_dot_threshold, 'nBins':str(nBins), 'reward':'without limite theta dot'}

    
    model = {} # model is a dictionary of dictionaries, which maps states to actions to
                        # (reward, next_state) tuples
    

    import time
    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")
    logpath = '/Users/lfu/Documents/IBRID/0_Script/cart_pole/q_learning/results/DynaQ_'+dt_string
    os.makedirs(logpath, exist_ok=True)
    os.chdir(logpath)
    with open('0_INFO.json', 'w') as file:
        json.dump(INFO, file, indent=4)

    bins = create_bins(x_threshold, theta_dot_threshold, nBins=nBins)
    result, Q, model = play_many_episodes(observationNum, actionNum, nBins, numEpisode, min_epsilon, min_lr, model)
    print('finished')
    
    result.to_csv('DynaQ_learning_curve_alpha0_'+str(ALPHA0)+'_gamma_'+str(GAMMA)+'_minEps_'+str(min_epsilon)+'.csv')
    Q.T.to_csv('DynaQ_Qtable_alpha0_'+str(ALPHA0)+'_gamma_'+str(GAMMA)+'_minEps_'+str(min_epsilon)+'.csv')
#%%