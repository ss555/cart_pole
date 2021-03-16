'''
FORMAT
a v u
'''
import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import glob
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('./'))

def parce_csv(absPath):
    data    =np.zeros(shape=(3,))
    namesRaw=glob.glob(absPath+'./*.csv')
    namesRaw.sort()
    for filename in namesRaw:
        # fileData=pd.read_csv(filename)
        fileData=np.genfromtxt(filename, delimiter=',')
        processedData=preprocess_data(fileData[:,1:], False)
        data=np.vstack((data, processedData))
    return data[1:,:]
def preprocess_data(fileData,plot=False):
    ##PWM time(s) position(x)
    #TODO verify u
    u=-fileData[:,0]/255*12
    dt=np.mean(np.diff(fileData[:,1]))
    v=np.convolve(fileData[:,2],[0.5,0,-0.5],'valid')/dt
    # v=np.convolve(fileData[:,2],[1,-1],'valid')/dt
    a=np.convolve(fileData[:,2],[1,-2,1],'valid')/(dt**2)
    if plot:
        ax1 = plt.subplot(311)
        plt.plot(fileData[1:-1,1], u)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_ylabel('Position in m', fontsize=5)
        ax2 = plt.subplot(312,sharex=ax1)
        plt.plot(fileData[1:-1,1], v)
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set_ylabel('Speed (m/s)',fontsize=5)
        # share x
        ax3 = plt.subplot(313, sharex=ax2)
        plt.plot(fileData[1:-1,1], a)
        ax3.set_ylabel('acceleration (m/s^2)',fontsize=5)
        ax3.set_xlabel('time in s')
        plt.show()
    return np.vstack([a,v,u[1:-1]]).T
def integrate_acceleration(a,b,c,u,timeArr):
    v=np.zeros(shape=(len(timeArr)))
    for i in range(1,len(timeArr)):
        dt=timeArr[i]-timeArr[i-1]
        dv=a*v[i-1]+b*u[i-1]+c*np.sign(v[i-1])
        v[i]=v[i-1]+dv*dt
    return v
def regression_chariot(data):
    #acc=vit*a+b*U+c*np.sign(vit)
    # np.random.shuffle(data)
    regB=data[:,0].reshape(-1,1) #acceleration
    regA=np.stack([data[:,1],data[:,2],np.sign(data[:,1])],axis=1)
    X=np.linalg.lstsq(regA,regB,rcond=None)[0]
    error=0.1
    return np.hstack([np.squeeze(X, axis=1), error])
def plot_experimental_fitted(absPath,fA,fB,fC):
    # dv = -a * vs[i] + b * u + c * np.sign(vs[i])
    names=glob.glob(absPath+'./*.csv')
    names.sort()
    for filename in names:
        # fileData=pd.read_csv(filename)
        fileData=np.genfromtxt(filename, delimiter=',')
        processedData=preprocess_data(fileData[:,1:], False)

        plt.rcParams['figure.dpi']=300
        fig1=plt.plot()
        # a=expData[:,0]
        u=processedData[:,2]
        v=processedData[:,1]

        v_fitted=integrate_acceleration(fA,fB,fC,u,timeArr=fileData[1:-1,2])
        plt.plot(fileData[1:-1,2],v,'.',label=("%.1fV" % u[0]))
        plt.plot(fileData[1:-1,2],v_fitted)
        plt.plot()
    plt.legend()
    plt.xlabel("Time, s")
    plt.ylabel("Velocity, m/s")
    plt.grid(True)
    plt.show()

print(os.getcwd())
absPath='/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/data/'
#pwm speed acceleration
expData=parce_csv(absPath)
#
[fA,fB,fC,error] = regression_chariot(expData)
plot_experimental_fitted(absPath,fA,fB,fC)

print(len(expData))
print(error)
print(f'{fA,fB,fC}')
#(-9.992699476436576, 0.5283959730526665, -0.4335098068332604)