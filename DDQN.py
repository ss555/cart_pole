#!/usr/bin/env python
# coding: utf-8

# # Double Deep Q-Network (DDQN)
# ---
# In this notebook, you will implement a DQN agent with OpenAI Gym's LunarLander-v2 environment.
# 
# ### 1. Import the Necessary Packages

# In[1]:


import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from env_custom import CartPoleCosSinTensionD
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
import plotly.express as px

# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("current dir", current_dir)


# ### 2. Instantiate the Environment and Agent
# 
# Initialize the environment in the code cell below.

# In[3]:


goal_X = 3
env = CartPoleCosSinTensionD()#
env = env.unwrapped
# env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


# Please refer to the instructions in `Deep_Q_Network.ipynb` if you would like to write your own DQN agent.  Otherwise, run the code cell below to load the solution files.

# In[5]:


from dqn_agent import Agent

agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)

# modelPath = Path(current_dir)
# modelList = modelPath.glob('*.pth')
modelFile = 'tmp'
# agent.qnetwork_local.load_state_dict(torch.load(modelFile))
# agent.qnetwork_target.load_state_dict(torch.load(modelFile))

# watch an untrained agent
state = env.reset()
# for j in range(200):
#     action = agent.act(state)
#     env.render(j)
#     state, reward, done, _ = env.step(action)
#     if done:
#         break 
# code, line_no = inspect.getsourcelines(env.reward)
# rewardFun = code[2].strip()
INFO = {
    'EnvName': env.__class__.__name__
        }
env.close()


now = datetime.now()
dt_string = now.strftime("%m%d_%H%M%S")
os.chdir('./tmp')
dirname = 'DDQN_'+dt_string
try:
    os.mkdir(dirname)
except:
    pass
os.chdir(dirname)


# ### 3. Train the Agent with DQN
# 
# Run the code cell below to train the agent from scratch.  You are welcome to amend the supplied values of the parameters in the function, to try to see if you can get better performance!
# 
# Alternatively, you can skip to the next step below (**4. Watch a Smart Agent!**), to load the saved model weights from a pre-trained agent.

# In[ ]:


n_episodes=25000
max_t=2000
eps_start=1.0
eps_end=0.01
eps_decay=0.999

def dqn(n_episodes=20000, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.9998):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    INFO.update({'n_episodes':n_episodes, 'max_t':max_t, 'eps_start':eps_start, 'eps_end':eps_end, 'eps_decay':eps_decay})
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=500)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    checkpoint_list = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 500 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if i_episode % 1000 == 0:
            detail = {'episode':i_episode+1, 'state_dict': agent.qnetwork_local.state_dict(), 'optimizer': agent.optimizer.state_dict(), }
            torch.save(detail, 'episode_'+str(i_episode)+'.pth')
        # if np.mean(scores_window)>=1900.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-500, np.mean(scores_window)))
        #     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_'+str(i_episode-500)+'.pth')
        #     checkpoint_list.append(i_episode-500)
            #break
    torch.save(agent.qnetwork_local.state_dict(), 'finalpoint.pth')
    return scores

import time
start_time = time.time()
scores = dqn(n_episodes, max_t, eps_start, eps_end, eps_decay)
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
INFO.update({'elapsed time':int(elapsed_time)})

## plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()
#%%
rewardsSeries = pd.Series(scores)
rewardsSeries.to_csv('scores.csv')
measure = rewardsSeries[:]
rollmean = measure.rolling(1000).mean()
rollstd = measure.rolling(1000).std()

fig = px.line(rewardsSeries, x=rewardsSeries.index, y=rewardsSeries.values)
fig.write_html('1_episode_reward'+'.html')
fig2 = px.line(rollmean, x=rollmean.index, y=rollmean.values)
fig2.write_html('1_rollmean.html')
fig3 = px.line(rollstd, x=rollstd.index, y=rollstd.values)
fig3.write_html('1_rollstd.html')

with open('0_INFO.json', 'w') as outfile:
    json.dump(INFO, outfile, indent=4)


#### 4. Watch a Smart Agent!
# In the next code cell, you will load the trained weights from file to watch a smart agent!
# In[28]:


# load the weights from file
from bokeh.plotting import figure, output_file, show, save
n_episodes=20000
for i in range(1, n_episodes+1):
    if i % 1000 == 0:
        checkpoint = torch.load('episode_'+str(i)+'.pth')
        #agent.qnetwork_local.load_state_dict(torch.load('episode_'+str(i)+'.pth'))
        agent.qnetwork_local.load_state_dict(checkpoint['state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        state = env.reset()
        for j in range(2000):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            if done:
                break 
        df = pd.DataFrame.from_dict(env.dList)
        fig = env.plot_df(df, 'timestep', ['x', 'x_dot', 'alpha', 'alpha_dot','action'], title='obs_epi_'+str(i))
        save(fig)

# for i in range(len(checkpoint_list)):
#     agent.qnetwork_local.load_state_dict(torch.load('checkpoint_'+str(checkpoint_list[i])+'.pth'))
#     state = env.reset()
#     for j in range(2000):
#         action = agent.act(state)
#         state, reward, done, _ = env.step(action)
#         if done:
#             break 
#     df = pd.DataFrame.from_dict(env.dList)
#     fig = env.plot_df(df, 'timestep', ['x', 'x_dot', 'alpha', 'alpha_dot','action'], title='checkpoint_'+str(checkpoint_list[i]))
#     save(fig)
env.close()


# ### 5. Explore
# 
# In this exercise, you have implemented a DQN agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
# - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task with discrete actions!
# - You may like to implement some improvements such as prioritized experience replay, Double DQN, or Dueling DQN! 
# - Write a blog post explaining the intuition behind the DQN algorithm and demonstrating how to use it to solve an RL environment of your choosing.  

# In[34]:


list(range(1,2))


# In[30]:


elapsed_time


# In[31]:


elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


# In[32]:


elapsed_time


# In[ ]:




