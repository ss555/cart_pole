'''
FORMAT
a v u
'''
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
from scipy import signal
sys.path.append(os.path.abspath('./'))
import plotly.express as px
from bokeh.palettes import d3
#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 100
colorPalette = d3['Category20'][20]
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
    try:
        figSave,ax = plt.subplots()
        legs=[]
        fileData = np.genfromtxt(filename, delimiter=',').T
        pwmStart = int(min(abs(fileData[:, 0])))
        pwmEnd   = int(max(fileData[:, 0]))
        stableSpeeds = []
        # plt.figure(figsize=(30,12),dpi=200)
        c=0
        samplePoints = 25
        fig = px.scatter(x=[0],y=[0])
        TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
        for i in range(pwmStart, pwmEnd + 10, 10):
            # try:
            localData = fileData[fileData[:, 0] == i, :]
            dt = np.mean(np.diff(localData[:, 1]))
            v = np.convolve(localData[:, 2], [0.5, 0, -0.5], 'valid') / dt
            a = np.convolve(localData[:, 2], [1, -2, 1], 'valid') / (dt ** 2)
            u = [((i) / 255 * 12)]
            time_offset = localData[0,1]
            v_fitted = integrate_acceleration(fA,fB,fC,fD,u, timeArr=localData[:,1])
            if applyFiltering:
                #butterworth filtering
                bf, af = signal.butter(Nf, 2 * (dt * fc))
                v = signal.filtfilt(bf, af, v, padtype=None)
            fig.add_scatter(x=localData[1:-1, 1], y=v, name=("%.1fV"%u[0]))
            fig.add_scatter(x=localData[:,1], y=v_fitted)
            stableSpeeds.append(np.min(v))
            if abs(i)%50==0 and abs(i)<210:
                ax.plot(localData[1:-1, 1]-localData[1, 1],v,'.',color = colorPalette[c])
                legs.append(str(round(-i/255*12,1))) # - because of inverted tension
                legs.append(str(round(-i / 255 * 12, 1))+' fitted') # - because of inverted tension
                ax.plot(localData[:,1]-localData[0,1], v_fitted,color = colorPalette[c+5])
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
            v_fitted = integrate_acceleration(fA, fB, fC,fD,u, timeArr=localData[:, 1])
            if applyFiltering:
                bf, af = signal.butter(Nf, 2 * (dt * fc))
                v = signal.filtfilt(bf, af, v, padtype=None)
            try:
                print(f'bf: {bf}, af: {af}')
            except:
                pass
            fig.add_scatter(x=localData[1:-1, 1], y=v, name=("%.1fV"%u[0]))
            fig.add_scatter(x=localData[:, 1], y=v_fitted)
            stableSpeeds.append(np.max(v))
            if abs(i)%50==0 and abs(i)<210:
                ax.plot(localData[1:-1, 1]-localData[1, 1], v,'.', color = colorPalette[c])
                ax.plot(localData[:,1]-localData[0,1], v_fitted, color = colorPalette[c+5])
                legs.append(str(round(i/255*12,1)))
                legs.append(str(round(i / 255 * 12, 1))+' fitted')
                c+=1
        fig.show()
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
        ax2.plot(tensions, stableSpeeds, 'o')
        ax2.set_xlabel('tension in [V]')
        ax2.set_ylabel('stable speed in [m/s]')
        ax2.grid()
        fig2.show()
        fig2.savefig('./EJPH/plots/regression_u_v.pdf')
    except:
        print('s')


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
#c++
# (-9.992699476436576, 0.5283959730526665, -0.4335098068332604)
# (-18.03005925191054, 0.965036433340654, -0.8992003750802359)

