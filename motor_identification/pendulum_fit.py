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
APPLY_FILTER = 'butterworth'#'ukf'#
absPath = '/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/idenPenduleCsv/angle_iden_alu.csv'
def integrate_theta_params(k, w, theta_ini, thetaDotIni, dt, steps=3000):
    #thetaDD=w*theta_dot-k*theta
    thetaDD = np.zeros(steps)
    thetaDot = np.zeros(steps)
    theta = np.zeros(steps)
    theta[0] = theta_ini
    thetaDot[0] = thetaDotIni
    # theta[0]
    for i in range(1, steps):
        thetaDD[i] = - w * np.sin(theta[i-1]) - k*thetaDot[i-1]
        thetaDot[i] = thetaDot[i-1] + dt * thetaDD[i]
        theta[i] = theta[i-1] + dt * thetaDot[i]
    #debug
    # fig,ax =plt.subplots(ncols=2)
    # ax[0].plot(theta)
    # # ax.show()
    # ax[1].plot(thetaDD)
    # fig.show()
    return theta


def integrate_theta_acc(theta_acc,theta_ini,thetaDotIni, timeArr):
    theta=np.zeros(shape=(len(theta_acc)))
    thetaDot=np.zeros(shape=(len(theta_acc)))
    theta[0] = theta_ini
    thetaDot[0] = thetaDotIni
    for i in range(1,len(theta_acc)):
        dt=timeArr[i]-timeArr[i-1]
        thetaDot[i] = thetaDot[i-1] + dt*theta_acc[i-1]
        theta[i] = theta[i-1] + dt*thetaDot[i]
    return theta

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
    # since theta=0 is up in the acquisition script, we have substracted pi
    data[:, 2] = data[:,2] - np.pi
    regA = np.stack((-np.sin(data[:,2]),data[:,1]),axis=1)
    # regA = np.stack((np.sin(data[:,2]),data[:,1]),axis=1)
    X = np.linalg.lstsq(regA,regB, rcond=None)[0]

    indStartPlot = 0 #important to have index to 0 when integrating, otherwise will be offset while integrating accel different from 0
    indEndPlot = 4000
    # indStartPlot = 2000
    # indEndPlot = 4000
    #plot fitted curve
    fig, ax = plt.subplots()
    ax.plot(time[indStartPlot:indEndPlot]-time[indStartPlot],regB[indStartPlot:indEndPlot],'r.') #plot real acceleration
    regRes = np.matmul(regA,X)
    #plot fitted curve acceleration
    ax.plot(time[indStartPlot:indEndPlot]-time[indStartPlot], regRes[indStartPlot:indEndPlot])
    ax.legend(['experimental','fitted'],loc='best')#bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('time in [s]')
    ax.set_ylabel('acceleration in [m/s^2]')
    ax.grid()
    ax.legend(['experimental', 'fitted'], loc='best')  # bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('time in [s]')
    ax.set_ylabel('acceleration in [rad/s^2]')
    fig.savefig('./EJPH/plots/regression_ddtheta_t.pdf')
    fig.show()

    # plot fitted curve ANGLE
    fig2, ax2 = plt.subplots()
    offsetBeginning=0#indEndPlot/2
    dt = np.mean(np.diff(time)[np.diff(time) < 0.1])#without reset phase np.diff(time) < 0.1
    # theta=integrate_theta_acc(regRes[indStartPlot:indEndPlot], theta_ini=data[indStartPlot,-1],thetaDotIni=data[indStartPlot,1], timeArr=time)
    #solve inegral equation
    theta = integrate_theta_params(w = X[0], k = X[1], theta_ini = data[indStartPlot,-1], thetaDotIni = data[indStartPlot,1], dt=dt,steps=(indEndPlot-indStartPlot))
    ax2.plot(time[indStartPlot:indEndPlot] - time[indStartPlot], data[indStartPlot+offsetBeginning:indEndPlot,-1], 'r.') #plot
    ax2.plot(time[indStartPlot:indEndPlot] - time[indStartPlot], theta[offsetBeginning:], 'b') #plot
    ax2.legend(['experimental', 'fitted'], loc='best')  # bbox_to_anchor=(1.05, 1))
    ax2.set_xlabel('time in [s]')
    ax2.set_ylabel('theta in [rad]')
    ax2.grid()
    fig2.tight_layout()
    fig2.savefig('./EJPH/plots/regression_theta_t.pdf')
    fig2.show()


    # ax.set_xlabel('time in [ms]')
    # ax.legend(['filtered experimental acceleration','fitted curve'],bbox_to_anchor=(1.05, 1))
    fig.show()
    return X


expData, time = process_angle_raw(absPath,plot=False)
# [wSquare,kViscous,cStatic]=fit_params(expData)
[wSquare, kViscous] = fit_params(expData,time=time)
print([wSquare, kViscous])
print(f'freq{np.sqrt(wSquare)}')
print(f'R - {9.806/wSquare}')

