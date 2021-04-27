import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import glob
import pandas as pd
import sys
import os
import math
sys.path.append(os.path.abspath('./'))
from scipy import signal
from plotly.subplots import make_subplots
import plotly.graph_objects as go
APPLY_FILTER=False
absPath='/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/idenPenduleCsv/angle_iden.csv'
def process_angle_raw(absPath,plot=True):
    data=np.genfromtxt(absPath,delimiter=',')
    #time position(in degree)
    aArr=[]
    vArr=[]
    posArr=[]
    timeArr=[]
    for i in range(int(max(data[0,:]))+1):
        idx=data[0,:] == i
        releventData = data[:,idx]
        time = releventData[1, :]
        posRaw = releventData[2, :] / 180 * math.pi
        dt = np.mean(np.diff(time))
        v = np.convolve(posRaw, [-0.5, 0, 0.5], 'valid') / dt
        a = np.convolve(posRaw, [1, -2, 1], 'valid') / dt ** 2
        # filter the signals
        if APPLY_FILTER:
            # frequency and order of the butterworth filter used in smoothing
            # 50ms
            fc = 4
            Nf = 4
            cutStart = 2 * Nf + 1  # cutting the edge effects of butterworth
            bf, af = signal.butter(Nf, 2 * (dt * fc))  # [1],[1,0] = no filter
            a = signal.filtfilt(bf, af, a, padtype=None)
            v = signal.filtfilt(bf, af, v, padtype=None)
            pos = signal.filtfilt(bf, af, posRaw, padtype=None)
            #CUT BORD EFFECTS
            # pos = pos[1+cutStart:-cutStart-1]
            # v = v[cutStart:-cutStart]
            # a = a[cutStart:-cutStart]
            # time = time[1+cutStart:-cutStart-1]
            #withous
            pos=pos[1:-1]
            time=time[1:-1]
            ##
        else:
            pos = posRaw[1:-1]
            time = time[1:-1]
        if i==1:
            k=1500
            a=a[:k]
            v=v[:k]
            time=time[:k]
            pos=pos[:k]
        if plot:
            fig=make_subplots(rows=3, cols=1)
            fig.add_trace(go.Scatter(x=time, y=pos, mode="lines"))
            fig.add_trace(go.Scatter(x=time, y=v), row=2, col=1)  # ,label='Xd')
            fig.add_trace(go.Scatter(x=time, y=a), row=3, col=1)  # ,label='XdF')
            fig.show()
        else:
            ax1 = plt.subplot(311)
            plt.plot(time, pos, 'g', label='position')
            ax1.legend(loc="upper right")
            ax2 = plt.subplot(312, sharex=ax1)
            plt.plot(time, v, label='velocity')
            ax2.legend(loc="upper right")
            ax3 = plt.subplot(313, sharex=ax1)
            ax3.legend(loc="upper right")
            plt.plot(time, a, 'r', label='acc')
            plt.show()
        aArr=np.hstack((aArr,a))
        vArr=np.hstack((vArr,v))
        posArr=np.hstack((posArr,pos))
        timeArr=np.hstack((timeArr,time))
    return np.vstack((aArr,vArr,posArr)).T

def fit_params(data):

    regB=data[:,0]
    # regA=np.stack((np.sin(data[:,2]),data[:,1],np.sign(data[:,1])),axis=1) #with static friction
    regA=np.stack((np.sin(data[:,2]),data[:,1]),axis=1)
    # theta_acc=w*sin(theta)-k*theta_dot-c*np.sign(theta_dot) with static
    # theta_acc=w*sin(theta)-k*theta_dot #viscous Fr
    X=np.linalg.lstsq(regA,regB, rcond=None)[0]
    return X
expData = process_angle_raw(absPath)

# [wSquare,kViscous,cStatic]=fit_params(expData)
[wSquare,kViscous]=fit_params(expData)
print([wSquare,kViscous])
print(f'freq{np.sqrt(wSquare)}')
#1data
#[-1.4349472355174064, 0.07157002894573239, -0.013867312724331888]
#[21.274437162999224, 0.09899092543427149]
#many data [f 4.610673971440863, k 0.08495031673343573]