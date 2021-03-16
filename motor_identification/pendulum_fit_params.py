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


absPath='/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/idenCsv/angle_iden.csv'
def process_angle_raw(absPath,plot=True):
    data=np.genfromtxt(absPath,delimiter=',')
    #time position(in degree)
    time=data[1,:]
    posRaw=data[2,:]/180
    dt=np.mean(np.diff(time))
    v=np.convolve(posRaw,[-0.5,0,0.5],'valid')/dt
    cutStart=0
    cutEnd=10
    v=v[cutStart:]
    a=np.convolve(posRaw,[1,-2,1],'valid')/dt**2
    a=a[cutStart:]
    if plot:
        ax1 = plt.subplot(311)
        plt.plot(time[1+cutStart:-1], posRaw[1+cutStart:-1],'g',label='position')
        ax1.legend(loc="upper right")
        ax2=plt.subplot(312,sharex=ax1)
        plt.plot(time[1+cutStart:-1],v,label='velocity')
        ax2.legend(loc="upper right")
        ax3=plt.subplot(313,sharex=ax1)
        ax3.legend(loc="upper right")
        plt.plot(time[1+cutStart:-1],a,'r',label='acc')
        plt.show()
    return np.vstack((a,v,posRaw[1+cutStart:-1],time[1+cutStart:-1])).T

def fit_params(data):

    regB=data[:,0]
    regA=np.stack((np.sin(data[:,2]),data[:,1],np.sign(data[:,1])),axis=1)
    # theta_acc=w*sin(theta)-k*theta_dot-c*np.sign(theta_dot) with static
    # theta_acc=w*sin(theta)-k*theta_dot #viscous Fr
    X=np.linalg.lstsq(regA,regB, rcond=None)[0]
    return X
expData = process_angle_raw(absPath)

[w,kViscous,cStatic]=fit_params(expData)
print([w,kViscous,cStatic])
#[-1.4349472355174064, 0.07157002894573239, -0.013867312724331888]