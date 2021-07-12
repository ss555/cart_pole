import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
sys.path.append(os.path.abspath('./'))
from scipy import signal
from plotly.subplots import make_subplots
import plotly.graph_objects as go
'''
fitting parameters from pendulum free fall:
APPLY_FILTER can be in 3 modes: None, ukf or butterworth(recommended)
'''
APPLY_FILTER='butterworth'#'ukf'#
absPath = '/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/idenPenduleCsv/angle_iden_alu.csv'
def process_angle_raw(absPath,plot=True):
    '''

    :param absPath: path to .csv file
    :param plot: plot the data or not
    :return: aArr,vArr,posArr as AN ARRAY at every point in time
    '''
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
        #skip the beginning biased data
        posRaw = posRaw[10:]
        time = time[10:]
        dt = np.mean(np.diff(time))
        v = np.convolve(posRaw, [-0.5, 0, 0.5], 'valid') / dt
        a = np.convolve(posRaw, [1, -2, 1], 'valid') / dt ** 2
        # filter the signals
        if APPLY_FILTER=='butterworth':
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
            pos = pos[1+cutStart:-cutStart-1]
            v = v[cutStart:-cutStart]
            a = a[cutStart:-cutStart]
            time = time[1+cutStart:-cutStart-1]
            #without filter
            # pos=pos[1:-1]
            # time=time[1:-1]
            ##
        elif APPLY_FILTER=='ukf':
            def fx(x, dt):
                xout = np.empty_like(x)
                xout[0] = x[1] * dt + x[0]
                xout[1] = x[1]
                return xout
            def hx(x):
                return x[:1]  # return position [x]
            from numpy.random import randn
            from filterpy.kalman import UnscentedKalmanFilter
            from filterpy.common import Q_discrete_white_noise
            from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
            # sigmas = JulierSigmaPoints(n=2, kappa=1)#, alpha=.3, beta=2.)
            sigmas = MerweScaledSigmaPoints(n=2, kappa=1, alpha=.3, beta=2.)
            ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=dt, hx=hx, fx=fx, points=sigmas)
            ukf.P *= 1e-3
            ukf.R *= 1e-5
            ukf.Q = Q_discrete_white_noise(2, dt=dt, var=1.0)
            ukfPos=[]
            ukfSpeed=[]
            for i in range(len(posRaw)):
                ukf.predict()
                ukf.update(posRaw[i])
                ukfPos.append(ukf.x[0])
                if i>0:
                    ukfSpeed.append(-(ukfPos[-1]-ukfPos[-2])/dt)
            # ukfSpeed=
            ukfPos = ukfPos[1:-1]
        elif APPLY_FILTER=='None':
            print('no data filtering')

            pos = posRaw[1:-1]
            time = time[1:-1]
        if plot:
            if APPLY_FILTER=='ukf':
                pos = [p-np.pi for p in pos]
                ukfPos = [u-np.pi for u in ukfPos]
                figS, ax = plt.subplots(figsize=(6, 3))
                ax.plot(time,pos,'ro')
                ax.plot(time,ukfPos,'b')
                figS.savefig('./EJPH/plots/pendulum_ukf_filtering.pdf')
                figS.show()
            fig=make_subplots(rows=3, cols=1,shared_xaxes=True)
            fig.add_trace(go.Scatter(x=time, y=pos, mode='markers'))
            fig.add_trace(go.Scatter(x=time, y=v, mode="lines"), row=2, col=1)  # ,label='Xd')
            if APPLY_FILTER=='ukf':
                fig.add_trace(go.Scatter(x=time, y=ukfPos, mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=time, y=ukfSpeed, mode="lines"), row=2, col=1)

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
        try:
            cutEdgeInd=1
            aArr=np.hstack((aArr,a[cutEdgeInd:-cutEdgeInd]))
            vArr=np.hstack((vArr,v[cutEdgeInd:-cutEdgeInd]))
            posArr=np.hstack((posArr,pos[cutEdgeInd:-cutEdgeInd]))
            timeArr=np.hstack((timeArr,time[cutEdgeInd:-cutEdgeInd]))
        except:
            print('smth is wrong with the data')
    return np.vstack((aArr,vArr,posArr)).T , timeArr

def fit_params(data, time=None):
    '''

    :param data: aArr,vArr,posArr
    :return: regression coefficients (X: w**2,K(viscous)) that satisfy(minimises) the eq. : |b-ax|,
     where b is aArr(acceleration array) and a is [sin(theta), theta]
    '''
    regB = data[:,0]
    # since theta=0 is up in the acquisition script, we have an inverted sine in the equation ddot(theta) = w**2 * sin(theta) + Kvisc*theta)
    regA = np.stack((np.sin(data[:,2]),data[:,1]),axis=1)
    X = np.linalg.lstsq(regA,regB, rcond=None)[0]
    try:
        indStartPlot = 2000
        indEndPlot = 4000
        #plot fitted curve
        fig,ax=plt.subplots()
        ax.plot(time[indStartPlot:indEndPlot]-time[indStartPlot],regB[indStartPlot:indEndPlot],'r.')
        regRes=np.matmul(regA,X)
        ax.plot(time[indStartPlot:indEndPlot]-time[indStartPlot],regRes[indStartPlot:indEndPlot])

        ax.legend(['experimental','fitted'],loc='best')#bbox_to_anchor=(1.05, 1))
        ax.set_xlabel('time in [s]')
        ax.set_ylabel('acceleration in [m/s^2]')
        # ax.set_xlabel('time in [ms]')
        # ax.legend(['filtered experimental acceleration','fitted curve'],bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig('./EJPH/plots/regression_theta_ddot.pdf')
        fig.show()
    except:
        print('smth is wrong to plot fitted curve')
    return X

expData, time = process_angle_raw(absPath,plot=False)
# [wSquare,kViscous,cStatic]=fit_params(expData)
[wSquare,kViscous] = fit_params(expData,time=time)
print([wSquare,kViscous])
print(f'freq{np.sqrt(wSquare)}')

plot_fitted = False
if plot_fitted:
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
    figS, ax = plt.subplots(figsize=(6, 3))
    ax.plot(time, posRaw, 'ro')
    figS.savefig('./EJPH/plots/pendulum_ukf_filtering.pdf')
    figS.show()

    indStartPlot = 2000
    indEndPlot = 4000
    fig,ax=plt.subplots()
    ax.plot(regB[indStartPlot:indEndPlot],'r.')
    regRes=np.matmul(regA,X)
    ax.plot(regRes[indStartPlot:indEndPlot])

    ax.legend(['experimental','fitted'],loc='best')#bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('timesteps')
    ax.set_ylabel('acceleration in m/s^2')
    # ax.set_xlabel('time in [ms]')
    # ax.legend(['filtered experimental acceleration','fitted curve'],bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('./EJPH/plots/regression_theta_ddot.pdf')
    fig.show()
#1data
#[-1.4349472355174064, 0.07157002894573239, -0.013867312724331888]
#[21.274437162999224, 0.09899092543427149]
#many data [f 4.610673971440863, k 0.08495031673343573]