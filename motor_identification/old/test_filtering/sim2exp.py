import numpy as np
import sys
import os
sys.path.append(os.path.abspath('./motor_identification/test_filtering'))
sys.path.append(os.path.abspath('./../..'))
sys.path.append(os.path.abspath('~/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/test_filtering'))
sys.path.append(os.path.abspath('~/1-THESE/4-sample_code/1-DDPG/12-STABLE3/'))
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from env_custom import CartPoleDiscrete

print(os.path.abspath('./'))
[f_a,f_b,f_c,f_d]=[-20, 0.9, -0.6, -0.09352431017546964] #-0.09352431017546964/0.9/12*255=2.2
# [f_a,f_b,f_c,f_d]=[-8.85137934,  0.41453099, -0.87260858, -0.11265897] #-0.09352431017546964/0.9/12*255=2.2
env=CartPoleDiscrete(Te=0.05, randomReset=False, integrator='rk4',f_a=f_a,f_b=f_b,f_c=f_c,f_d=f_d)
obsArr = np.array(pd.read_csv("obs.csv"))
timeArr= np.array(pd.read_csv("time.csv")).squeeze(axis=1)
actArr = np.array(pd.read_csv("actArr.csv")).squeeze(axis=1)
rewardsArr=[]
obsArrSim=np.zeros_like(obsArr)
obsArrSim[0,:]=env.reset(xIni=0.0)
for i, action in enumerate(actArr):
    obs, rewards, dones, _ = env.step(int(action))
    rewardsArr.append(rewards)
    obsArrSim[i+1,:] = obs
    # time.sleep(Te)
    # env.render()
    if dones:
        break
fig = px.scatter(x=timeArr, y=obsArr[:, 0],   title='observations through time')
# fig.add_scatter(x=timeArr, y=obsArrSim[:, 0], name='X simulated')
fig.add_scatter(x=timeArr, y=np.arctan2(obsArr[:, 3],obsArr[:, 2]),    name='theta through time')
fig.add_scatter(x=timeArr, y=np.arctan2(obsArrSim[:, 3],obsArrSim[:, 2]), name='thetaSim through time')
fig.show()