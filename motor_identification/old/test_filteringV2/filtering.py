import numpy as np
import sys
import os
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./motor_identification/test_filtering'))
print(os.path.abspath('./motor_identification/test_filtering'))
from src.utils import plot
dt = 0.05
# 50ms
fc = 4
Nf = 4
obsArr = np.array(pd.read_csv("obs.csv"))
timeArr=np.array(pd.read_csv("time.csv"))
actArr=np.array(pd.read_csv("actArr.csv"))
# if FILTERING:
bf, af = signal.butter(Nf, 2 * (dt * fc))
# v = signal.filtfilt(bf, af, v, padtype=None)
filteredObs=np.zeros_like(obsArr)
lObs=np.zeros_like(obsArr)
for i in range(1,len(obsArr)):
    for j in range(len(obsArr[0])):
        filteredObs[:i,j] = signal.filtfilt(bf, af, obsArr[:i,j], padtype=None)
        lObs[:i,j] = signal.lfilter(bf, af, signal.lfilter(bf, af, obsArr[:i,j]))
#x x_d alpha alpha_d
dt=np.mean(np.diff(timeArr.squeeze(axis=1)))
v=np.array(obsArr[:,1])
regB=np.convolve(v,[1,-1],'valid')/dt#acceler
regA=np.stack([v,actArr.squeeze(axis=1),np.sign(v)],axis=1)
X=np.linalg.lstsq(regA[:-1],regB)[0]
print(f'{X}')
timeArr=timeArr.squeeze(axis=1)
fig = px.scatter(x=timeArr, y=filteredObs[:, 0], title='observations through time')
fig = px.scatter(x=timeArr, y=lObs[:, 0], title='observations through timel')
fig.add_scatter(x=timeArr, y=filteredObs[:, 1], name='x_dot')
fig.add_scatter(x=timeArr, y=lObs[:, 1], name='x_dotl')
# fig.add_scatter(x=timeArr, y=filteredObs[:, 4], name='theta_dot')
# fig.add_scatter(x=timeArr, y=actArr, name='actionThroughTime')
# fig.add_scatter(x=timeArr, y=filteredObs[:, 4], name='theta_dot')
fig.show()
'''
plt.plot(timeArr,actArr,'ro')
plt.show()
'''
def mapActTension(actArr):
    actArr=actArr-1
    actArr=actArr*180
    return actArr
actArr=mapActTension(actArr)
plt.plot(timeArr,actArr,'ro')
plt.show()

regA=np.array()