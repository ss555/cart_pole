import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
from scipy import signal
from bokeh.palettes import d3
sys.path.append(os.path.abspath('./'))
#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 100
colorPalette = d3['Category20'][20]
import plotly.express as px
SCALE = 2.5
# figArticle, axis = plt.subplots(nrows=3)#,figsize=(SCALE*3,SCALE*3.7125))
figArticle, axis = plt.subplots(nrows=3, figsize=(SCALE*3,SCALE*5.56875))

# figArticle, axis = plt.subplots(nrows=3,figsize=(SCALE*3,SCALE*6))
# figArticle, axis = plt.subplots(nrows=3,figsize=(SCALE*1.618,SCALE*3))
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

def process_angle_raw(absPath,plot=False):
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
    # axis[0].set_aspect('equal')#1.618)
    axis[0].plot(time[indStartPlot:indEndPlot] - time[indStartPlot], data[indStartPlot+offsetBeginning:indEndPlot,-1], 'r.')
    axis[0].plot(time[indStartPlot:indEndPlot] - time[indStartPlot], theta[offsetBeginning:], 'b')
    axis[0].set_xlabel('time in [s]')
    axis[0].set_ylabel('theta in [rad]')
    axis[0].legend(['experimental', 'fitted'], loc='best')
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
#CHARIOT
'''
FORMAT
a v u
'''


def parce_csv(absPath,PLOT=False,fitTensionMin=None, fitTensionMax=None, weightedStartRegression=0,weight=10):
    data     = np.zeros(shape=(3,))
    namesRaw = glob.glob(absPath+'/*.csv')
    namesRaw.sort()
    dt=None
    if namesRaw==[]:
        print(f'no files found on {absPath}')
        raise Exception

    for filename in namesRaw:
        # fileData=pd.read_csv(filename)
        fileData = np.genfromtxt(filename, delimiter=',')
        processedData,weightedData = preprocess_data(fileData[:,:].T, plot=PLOT, fitTensionMin=fitTensionMin, fitTensionMax=fitTensionMax,weightedStartRegression=weightedStartRegression,weight=weight)
        data = np.vstack((data, processedData))
        dt = np.mean(np.diff(data[:,2]))
    return data[1:,:],dt,weightedData

def preprocess_data(fileData,plot=False,weightedStartRegression=0,weight=10, fitTensionMin=None,fitTensionMax=None):
    '''
    ATTENTION: tension is inverted, i.e when POSITIVE PWM applied there is - speed,
    SO WE INVERT it when adding to final array
    :param fileData:
    :param plot:
    :param weightedStartRegression:
    :param weight:
    :param fitTensionMin:
    :param fitTensionMax:
    :return:
    '''
    ##PWM time(s) position(x)
    res=np.zeros_like(fileData)
    weightedRes=np.zeros_like(fileData)
    pwmStart=int(min(abs(fileData[:,0]))) if fitTensionMin==None else fitTensionMin
    pwmEnd=int(max(fileData[:,0])) if fitTensionMax==None else fitTensionMax
    cStart=0
    for i in range(pwmStart,pwmEnd+10,10):
        try:
            localData=fileData[fileData[:,0]==i,:]
            dt=np.mean(np.diff(localData[:,1]))
            v=np.convolve(localData[:,2],[0.5,0,-0.5],'valid')/dt
            a=np.convolve(localData[:,2],[1,-2,1],'valid')/(dt**2)
            res[cStart:(cStart+len(v)),:]=np.stack([a,v,np.ones(len(a))*i/255*12]).T
            if weightedStartRegression != 0:
                weightedRes[cStart:(cStart+len(v)),:]=res[cStart:(cStart+len(v)),:]
                weightedRes[cStart:cStart+weightedStartRegression]=weightedRes[cStart:cStart+weightedStartRegression] * weight

            cStart += len(a)
        except:
            print('conv error')
    for i in range(pwmStart,pwmEnd+10,10):
        try:
            localData=fileData[fileData[:,0]==-i,:]
            dt=np.mean(np.diff(localData[:,1]))
            v=np.convolve(localData[:,2],[0.5,0,-0.5],'valid')/dt
            a=np.convolve(localData[:,2],[1,-2,1],'valid')/(dt**2)
            res[cStart:(cStart + len(v)), :] = np.stack([a, v, np.ones(len(a)) * (-i) / 255 * 12]).T
            if weightedStartRegression != 0:
                weightedRes[cStart:(cStart+len(v)),:]=res[cStart:(cStart+len(v)),:]
                weightedRes[cStart:cStart+weightedStartRegression]=weightedRes[cStart:cStart+weightedStartRegression] * weight
            cStart += len(a)
        except:
            print('conv error')
    if plot:
        ax1 = plt.subplot(311)
        plt.plot(fileData[1:-1,1], localData[:,2])
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
    if weightedStartRegression==0:
        weightedRes=np.copy(res)
    return res, weightedRes
def integrate_acceleration(a,b,c,d,u,timeArr): # integrates acceleration with the dynamic model
    v=np.zeros(shape=(len(timeArr)))
    for i in range(1,len(timeArr)):
        dt=timeArr[i]-timeArr[i-1]
        # acc=vit*a+b*U+c*np.sign(vit)+d*sign(u)
        n=10
        vprev=v[i-1]
        for j in range(n):
            dv=a*vprev+b*u[0]+c*np.sign(vprev)+d
            v[i]=vprev+dv*dt/n
            vprev=v[i]
    return v

def regression_chariot(data,symmetricTension=True):
    '''
    performs the matrix inversion to determinse the parameters a,b,c,d of acc=vit*a+b*U+c*np.sign(vit)+d
    :param data: [N,3] acc|speed|tension
    :param weightedStartRegression:
    :param weight:
    :param symmetricTension:
    :return:
    '''
    regB=data[:,0].reshape(-1,1) #acceleration
    if symmetricTension:
        regA=np.stack([data[:,1],-data[:,2],np.sign(data[:,1])],axis=1)# -VOLTAGE because of inverted tension
    else:
        regA=np.stack([data[:,1],-data[:,2],np.sign(data[:,1]),np.ones_like(np.sign(-data[:,2]))],axis=1) # -VOLTAGE because of inverted tension
    X=np.linalg.lstsq(regA,regB,rcond=None)

    #ANOTHER METHOD USING RANSACRegressor
    # import sklearn.linear_model as sk
    # from sklearn.linear_model import SGDRegressor
    # from sklearn.pipeline import make_pipeline
    # from sklearn.preprocessing import StandardScaler
    # reg = sk.LinearRegression().fit(regA,regB)
    # reg2 = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    # reg2.fit(regA,regB)
    # from sklearn.linear_model import Ridge
    # clf = Ridge(alpha=1.0)
    # clf.fit(regA, regB)
    # from sklearn.linear_model import RANSACRegressor
    # reg = RANSACRegressor(random_state=0).fit(regA, regB)
    # reg.score(regA, regB)
    error=X[1]
    X=X[0]
    X=np.squeeze(X, axis=1)
    if symmetricTension: #pour standardiser les donnes fd=0
        X=np.hstack([X,0])
    return np.hstack([X, error])

def plot_experimental_fitted(filename,fA,fB,fC,fD,applyFiltering=False,Nf = 4,fc=4):
    # dv = -a * vs[i] + b * u + c * np.sign(vs[i])
    figSave,ax = plt.subplots()
    legs=[]
    fileData = np.genfromtxt(filename, delimiter=',').T
    pwmStart = int(min(abs(fileData[:, 0])))
    pwmEnd   = int(max(fileData[:, 0]))
    stableSpeeds = []
    # plt.figure(figsize=(30,12), dpi=200)
    TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
    c = 7
    offVoltage = 4
    # fig = px.scatter(x=[0],y=[0])
    for i in range(pwmStart, pwmEnd + 10, 10):
        # try:
        localData = fileData[fileData[:, 0] == i, :]
        dt = np.mean(np.diff(localData[:, 1]))
        v = np.convolve(localData[:, 2], [0.5, 0, -0.5], 'valid') / dt
        a = np.convolve(localData[:, 2], [1, -2, 1], 'valid') / (dt ** 2)
        u = [((i) / 255 * 12)]
        time_offset = localData[0,1]
        v_fitted = integrate_acceleration(fA,fB,fC,fD,u,timeArr=localData[:,1])
        if applyFiltering:
            #butterworth filtering
            bf, af = signal.butter(Nf, 2 * (dt * fc))
            v = signal.filtfilt(bf, af, v, padtype=None)
            #TODOO? cut the edges
        # fig.add_scatter(x=localData[1:-1, 1], y=v, name=("%.1fV"%u[0]))
        # fig.add_scatter(x=localData[:,1], y=v_fitted)
        stableSpeeds.append(np.min(v))

        if abs(i)%50==0 and abs(i)<210:
            ax.plot(localData[1:-1, 1]-localData[1, 1],v,'.')
            axis[1].plot(localData[1:-1, 1]-localData[1, 1],v,'.' ,color=colorPalette[c])
            legs.append(str(round(-i/255*12,1))) # - because of inverted tension
            legs.append(str(round(-i / 255 * 12, 1))+' fitted') # - because of inverted tension
            ax.plot(localData[:,1]-localData[0,1], v_fitted)
            axis[1].plot(localData[:,1]-localData[0,1], v_fitted ,color=colorPalette[c+offVoltage])
            c+=1

        # except:
        #     print('plot_experimental_fitted error')
    stableSpeeds.reverse()

    for i in range(pwmStart, pwmEnd + 10, 10):
        # try:
        localData = fileData[fileData[:, 0] == -i, :]
        dt = np.mean(np.diff(localData[:, 1]))
        v = np.convolve(localData[:, 2], [0.5, 0, -0.5], 'valid') / dt
        a = np.convolve(localData[:, 2], [1, -2, 1], 'valid') / (dt ** 2)
        u = [((-i) / 255 * 12)]
        v_fitted = integrate_acceleration(fA, fB, fC,fD, u, timeArr=localData[:, 1])
        if applyFiltering:
            bf, af = signal.butter(Nf, 2 * (dt * fc))
            v = signal.filtfilt(bf, af, v, padtype=None)
        try:
            print(f'bf: {bf}, af: {af}')
        except:
            pass
        # fig.add_scatter(x=localData[1:-1, 1], y=v, name=("%.1fV"%u[0]))
        # fig.add_scatter(x=localData[:, 1], y=v_fitted)
        stableSpeeds.append(np.max(v))
        # stableSpeeds.append(np.mean(v[:-2]))
        if abs(i)%50==0 and abs(i)<210:
            ax.plot(localData[1:-1, 1]-localData[1, 1], v,'.')
            ax.plot(localData[:,1]-localData[0,1], v_fitted)
            legs.append(str(round(i/255*12,1)))
            legs.append(str(round(i / 255 * 12, 1))+' fitted')
            if i==50:
                axis[1].plot(localData[1:-1, 1]-localData[1, 1], v,'.',color=colorPalette[0])
            elif i == 100:
                axis[1].plot(localData[1:-1, 1] - localData[1, 1], v, '.', color=colorPalette[2])
            elif i == 150:
                axis[1].plot(localData[1:-1, 1] - localData[1, 1], v, '.', color=colorPalette[4])
            elif i == 200:
                axis[1].plot(localData[1:-1, 1] - localData[1, 1], v, '.', color=colorPalette[6])
            else:
                axis[1].plot(localData[1:-1, 1] - localData[1, 1], v, '.', color=colorPalette[c])
            axis[1].plot(localData[:,1]-localData[0,1], v_fitted,color=colorPalette[c+offVoltage])
            c += 1
    # axis[1].set_aspect(10)
    axis[1].legend(legs)
    axis[1].set_xlim([-0.5, 1.5])
    axis[1].set_xlabel('time in [s]')
    axis[1].set_ylabel('speed in [m/s]')
    # fig.show()
    ax.set_xlabel('time in [s]')
    ax.set_ylabel('speed in [m/s]')
    ax.set_xlim(left=0.0,right = 1.5)
    ax.grid()
    ax.legend(legs,loc = 'upper right',bbox_to_anchor=(1.35, 1.0))
    plt.tight_layout()
    figSave.savefig('./EJPH/plots/regression_chariot.pdf')
    figSave.show()
    fig2,ax2 = plt.subplots()
    tensions = np.hstack(([-i for i in range(pwmEnd, pwmStart-1, -10)],[i for i in range(pwmStart, pwmEnd + 10, 10)]))
    ax2.plot(tensions,stableSpeeds,'o')
    ax2.set_xlabel('tension in [V]')
    ax2.set_ylabel('stable speed in [m/s]')
    ax2.grid()
    fig2.show()
    fig2.savefig('./EJPH/plots/regression_u_v.pdf')
    ax.plot(localData[1:-1, 1] - localData[1, 1], v, '.')
    ax.plot(localData[:, 1] - localData[0, 1], v_fitted)

    axis[2].plot(tensions,stableSpeeds,'o')
    axis[2].set_xlabel('tension in [V]')
    axis[2].set_ylabel('stable speed in [m/s]')
    axis[2].grid()
    axis[1].grid()
    axis[0].grid()
    coords = [-0.1, 0.95]
    axis[0].text(coords[0], coords[1], chr(97) + ')', transform=axis[0].transAxes, fontsize='x-large')
    axis[1].text(coords[0], coords[1], chr(98) + ')', transform=axis[1].transAxes, fontsize='x-large')
    axis[2].text(coords[0], coords[1], chr(99) + ')', transform=axis[2].transAxes, fontsize='x-large')

print(os.getcwd())
absPath='/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/idenChariotCsv'# absPath='/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/chariot_data'#+'./chariot_iden.csv'
#pwm speed acceleration

#processed:
#acceleration speed pwm
# expData,dt =parce_csv(absPath,False,None,None)
expData, dt, weightedData = parce_csv(absPath,weightedStartRegression=0,weight=200,fitTensionMin=50,fitTensionMax=190) #in practice fitTensionMax<200
# (-19.355136863835682, 0.925594504005501, 0.15323233104506603, -0.19643065915299515)

# weighted_data
[fA,fB,fC,fD,error] = regression_chariot(weightedData,symmetricTension=False)
plot_experimental_fitted(absPath+'/chariot_iden_alu.csv', fA, fB, fC, fD, applyFiltering=False, Nf = 4,fc = 2)

print(len(expData))
print(error)
print(f'{fA,fB,fC,fD}')
plt.tight_layout()
figArticle.savefig('./EJPH/plots/reg_all.pdf')
figArticle.show()

#c++
# (-9.992699476436576, 0.5283959730526665, -0.4335098068332604)
# (-18.03005925191054, 0.965036433340654, -0.8992003750802359)

