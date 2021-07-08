'''
FORMAT
a v u
'''
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
sys.path.append(os.path.abspath('./'))


def parce_csv(absPath,PLOT=False,fitTensionMin=None,fitTensionMax=None):
    data    = np.zeros(shape=(3,))
    namesRaw=glob.glob(absPath+'/*.csv')
    namesRaw.sort()
    dt=None
    for filename in namesRaw:
        # fileData=pd.read_csv(filename)
        fileData=np.genfromtxt(filename, delimiter=',')
        processedData=preprocess_data(fileData[:,:].T, plot=PLOT, fitTensionMin=fitTensionMin, fitTensionMax=fitTensionMax)
        data=np.vstack((data, processedData))
        dt=np.mean(np.diff(data[:,2]))
    return data[1:,:],dt
def preprocess_data(fileData,plot=False, fitTensionMin=None,fitTensionMax=None):
    ##PWM time(s) position(x)
    aArr=[]
    vArr=[]
    uArr=[]
    pwmStart=int(min(abs(fileData[:,0]))) if fitTensionMin==None else fitTensionMin
    pwmEnd=int(max(fileData[:,0])) if fitTensionMax==None else fitTensionMax
    for i in range(pwmStart,pwmEnd+10,10):
        try:
            localData=fileData[fileData[:,0]==i,:]
            dt=np.mean(np.diff(localData[:,1]))
            v=np.convolve(localData[:,2],[0.5,0,-0.5],'valid')/dt
            a=np.convolve(localData[:,2],[1,-2,1],'valid')/(dt**2)
            aArr=np.hstack((aArr,a))
            vArr=np.hstack((vArr,v))
            uArr=np.hstack((uArr,np.ones(len(a))*i/255*12))

        except:
            print('conv error')
    for i in range(pwmStart,pwmEnd+10,10):
        try:
            localData=fileData[fileData[:,0]==-i,:]
            dt=np.mean(np.diff(localData[:,1]))
            v=np.convolve(localData[:,2],[0.5,0,-0.5],'valid')/dt
            a=np.convolve(localData[:,2],[1,-2,1],'valid')/(dt**2)
            aArr=np.hstack((aArr,a))
            vArr=np.hstack((vArr,v))
            uArr=np.hstack((uArr,np.ones(len(a))*(-i)/255*12))
        except:
            print('conv error')
    if plot:
        ax1 = plt.subplot(311)
        plt.plot(fileData[1:-1,1], uArr[0])
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
    return np.vstack([aArr,vArr,uArr]).T
def integrate_acceleration(a,b,c,d,u,timeArr):
    v=np.zeros(shape=(len(timeArr)))
    for i in range(1,len(timeArr)):
        dt=timeArr[i]-timeArr[i-1]
        # acc=vit*a+b*U+c*np.sign(vit)+d*sign(u)
        dv=a*v[i-1]+b*u[0]+c*np.sign(v[i-1])+d
        v[i]=v[i-1]+dv*dt
    return v
def regression_chariot(data,weightedStartRegression=0,weight=10,symmetricTension=True):
    '''
    performs the matrix inversion to determinse the parameters a,b,c,d of acc=vit*a+b*U+c*np.sign(vit)+d
    :param data: [N,3] acc|speed|tension
    :param weightedStartRegression:
    :param weight:
    :param symmetricTension:
    :return:
    '''
    if weightedStartRegression != 0:
        data[:weightedStartRegression,:] = data[:weightedStartRegression,:] * weight

    regB=data[:,0].reshape(-1,1) #acceleration
    if symmetricTension:
        regA=np.stack([data[:,1],data[:,2],np.sign(data[:,1])],axis=1)
    else:
        regA=np.stack([data[:,1],data[:,2],np.sign(data[:,1]),np.ones_like(np.sign(data[:,1]))],axis=1)
    X=np.linalg.lstsq(regA,regB,rcond=None)[0]
    X=np.squeeze(X, axis=1)
    if symmetricTension: #pour standardiser les donnes fd=0
        X=np.hstack([X,0])
    error=0.1
    return np.hstack([X, error])
def plot_experimental_fitted(filename,fA,fB,fC,fD):
    # dv = -a * vs[i] + b * u + c * np.sign(vs[i])
    fileData=np.genfromtxt(filename, delimiter=',').T
    pwmStart = int(min(abs(fileData[:, 0])))
    pwmEnd = int(max(fileData[:, 0]))
    # plt.figure(figsize=(30,12),dpi=200)
    import plotly.express as px
    fig = px.scatter(x=[0],y=[0])
    for i in range(pwmStart, pwmEnd + 10, 10):
        # try:
        localData = fileData[fileData[:, 0] == i, :]
        dt = np.mean(np.diff(localData[:, 1]))
        v = np.convolve(localData[:, 2], [0.5, 0, -0.5], 'valid') / dt
        a = np.convolve(localData[:, 2], [1, -2, 1], 'valid') / (dt ** 2)
        u=[((i) / 255 * 12)]
        time_offset=localData[0,1]
        v_fitted = integrate_acceleration(fA,fB,fC,fD,u,timeArr=localData[:,1])
        fig.add_scatter(x=localData[1:-1, 1], y=v, name=("%.1fV"%u[0]))
        fig.add_scatter(x=localData[:,1], y=v_fitted)
        # except:
        #     print('plot_experimental_fitted error')
    for i in range(pwmStart, pwmEnd + 10, 10):
        # try:
        localData = fileData[fileData[:, 0] == -i, :]
        dt = np.mean(np.diff(localData[:, 1]))
        v = np.convolve(localData[:, 2], [0.5, 0, -0.5], 'valid') / dt
        a = np.convolve(localData[:, 2], [1, -2, 1], 'valid') / (dt ** 2)
        u=[((-i) / 255 * 12)]
        v_fitted = integrate_acceleration(fA, fB, fC,fD, u, timeArr=localData[:, 1])
        fig.add_scatter(x=localData[1:-1, 1], y=v, name=("%.1fV"%u[0]))
        fig.add_scatter(x=localData[:, 1], y=v_fitted)
    fig.show()

print(os.getcwd())
absPath='/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/chariot_data'#+'./chariot_iden.csv'
# absPath='/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/chariot_data_180PWM'
# absPath='/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/chariot_data_150_180PWM'
#initial data:
#pwm speed acceleration

#acceleration speed pwm
# expData,dt =parce_csv(absPath,False,None,None)
expData,dt = parce_csv(absPath,fitTensionMin=150,fitTensionMax=180) #-22.713789110751794, 1.0325247560625972, 0.5808162775799824
# (-19.355136863835682, 0.925594504005501, 0.15323233104506603, -0.19643065915299515)

# weighted_data
# [fA,fB,fC,error] = regression_chariot(expData)
[fA,fB,fC,fD,error] = regression_chariot(expData,symmetricTension=False)
# [fA,fB,fC,fD,error] = regression_chariot(expData,weightedStartRegression=3,weight=150,symmetricTension=False)
expData,dt = parce_csv(absPath,fitTensionMin=170,fitTensionMax=190)#,fitTensionMin=150,fitTensionMax=170)
[fA,fB,fC,fD] = [-21.30359185798466, 1.1088617953891196, -0.902272006611719, -0.03935160774012411]
plot_experimental_fitted(absPath+'/chariot_iden.csv',fA,fB,fC,fD)

print(f'sampling points: {len(expData)}')
print(f'error: {error}')
print(f'{fA,fB,fC,fD}')
#c++
# (-9.992699476436576, 0.5283959730526665, -0.4335098068332604)
# (-18.03005925191054, 0.965036433340654, -0.8992003750802359)


