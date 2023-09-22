import numpy as np
import sys
import os
import iir_filter
from scipy import signal
from scipy.fft import fft,fftfreq
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import sys
import os
from src.env_custom import CartPoleDiscreteButter,CartPoleDiscrete
sys.path.append(os.path.abspath('./motor_identification/test_filtering'))
sys.path.append(os.path.abspath('./../..'))
sys.path.append(os.path.abspath('~/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/test_filtering'))
sys.path.append(os.path.abspath('~/1-THESE/4-sample_code/1-DDPG/12-STABLE3/'))
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


print(os.path.abspath('./'))
[f_a,f_b,f_c,f_d]=[-19.976538106642725, 1.0287320880446733, -0.9326363456754534, 0.035395644087165744] #-0.09352431017546964/0.9/12*255=2.2
# [f_a,f_b,f_c,f_d]=[-8.26588227, 0.4399867,  -0.75644167, 0.0] #-0.09352431017546964/0.9/12*255=2.2
# env=CartPoleDiscrete(Te=0.05, randomReset=False, integrator='ode',f_a=f_a,f_b=f_b,f_c=f_c,f_d=f_d)
env=CartPoleDiscreteButter(Te=0.05, randomReset=False, f_a=f_a,f_b=f_b,f_c=f_c,f_d=f_d)
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
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=timeArr,y=obsArr[:, 0], mode="lines"))#,label='Xd')
fig.add_trace(go.Scatter(x=timeArr,y=obsArrSim[:, 0]), row=1, col=1)#,label='XdF')
fig.add_trace(go.Scatter(x=timeArr,y=np.arctan2(obsArr[:, 3],obsArr[:, 2])), row=2, col=1)#,label='Xd')
fig.add_trace(go.Scatter(x=timeArr,y=np.arctan2(obsArrSim[:, 3],obsArrSim[:, 2])), row=2, col=1)#,label='XdF')
# fig.add_scatter(x=timeArr, y=obsArr[:, 1],    name='xDot through time')
# fig.add_scatter(x=timeArr, y=obsArrSim[:, 1], name='xDotSim through time')
# fig.add_scatter(x=timeArr, y=obsArrSim[:, 1], name='theta through time')
# theta=np.arctan2(observations[:, 3],observations[:, 2])
# fig.add_scatter(x=timeArr[:, 0], y=theta, name='theta through time')
fig.show()