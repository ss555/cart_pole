import numpy as np
import sys
import os
sys.path.append(os.path.abspath('./../..'))
sys.path.append(os.path.abspath('/home/sardor/1-THESE/4-sample_code/1-DDPG/12-STABLE3/motor_identification/test_filteringV2'))
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
obsArr = np.array(pd.read_csv("obs.csv"))
timeArr=np.array(pd.read_csv("time.csv"))
actArr=np.array(pd.read_csv("actArr.csv"))
print(os.path.abspath('./motor_identification/test_filtering'))
from src.utils import plot
obsArr=np.array(obsArr)
dt=np.mean(np.diff(timeArr.squeeze(axis=1)))
length =0.4611167818372032
g=9.806
masscart=0.44
masspole=0.06
sintheta=obsArr[:,3][1:-1]
costheta=obsArr[:,2][1:-1]
theta_dot=obsArr[:,4][1:-1]

xacc=np.convolve(obsArr[:,0],[1,-2,1],'valid')/(dt**2)
force=xacc*(masscart + masspole * sintheta ** 2) - masspole * g * sintheta * costheta + masspole * theta_dot ** 2 * sintheta * length
# thetaacc = wAngularIni ** 2 * sintheta + xacc / length * costheta - theta_dot * K2

#x x_d alpha alpha_d

v=np.array(obsArr[:,1])[1:-1]
regB=force/masscart#acceler
tension=(actArr.squeeze(axis=1)-1)*8.47
tension=tension[1:-1]
regA=np.stack([v,tension,np.sign(v),np.ones_like(v)],axis=1)
X=np.linalg.lstsq(regA,regB,rcond=None)[0]
print(f'{X}') #[-8.26588227  0.4399867  -0.75644167]
timeArr=timeArr.squeeze(axis=1)
# import numpy as np
# import sys
# import os
# sys.path.append(os.path.abspath('./'))
# sys.path.append(os.path.abspath('./motor_identification/test_filtering'))
# from scipy import signal
# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# obsArr = np.array(pd.read_csv("obs.csv"))
# timeArr=np.array(pd.read_csv("time.csv"))
# actArr=np.array(pd.read_csv("actArr.csv"))
# print(os.path.abspath('./motor_identification/test_filtering'))
# from src.utils import plot
# dt = 0.05
# # 50ms
# fc = 4
# Nf = 4
# # if FILTERING:
# bf, af = signal.butter(Nf, 2 * (dt * fc))
# # v = signal.filtfilt(bf, af, v, padtype=None)
# filteredObs=np.zeros_like(obsArr)
# lObs=np.zeros_like(obsArr)
# for i in range(1,len(obsArr)):
#     for j in range(len(obsArr[0])):
#         filteredObs[:i,j] = signal.filtfilt(bf, af, obsArr[:i,j], padtype=None)
#         lObs[:i,j] = signal.lfilter(bf, af, signal.lfilter(bf, af, obsArr[:i,j]))
#
# length =0.4611167818372032
# g=9.806
# masscart=0.44
# masspole=0.06
# # xacc = (force + masspole * g * sintheta * costheta - masspole * theta_dot ** 2 * sintheta * length) / (masscart + masspole * sintheta ** 2)
# force=xacc*(masscart + masspole * sintheta ** 2) - masspole * g * sintheta * costheta + masspole * theta_dot ** 2 * sintheta * length
# # thetaacc = wAngularIni ** 2 * sintheta + xacc / length * costheta - theta_dot * K2
#
# #x x_d alpha alpha_d
# dt=np.mean(np.diff(timeArr.squeeze(axis=1)))
# v=np.array(obsArr[:,1])
# regB=np.convolve(v,[1,-1],'valid')/dt#acceler
# tension=(actArr.squeeze(axis=1)-1)*8.47
# regA=np.stack([v,tension,np.sign(v)],axis=1)
# X=np.linalg.lstsq(regA[:-1],regB)[0]
# print(f'{X}')
# timeArr=timeArr.squeeze(axis=1)
# # fig = px.scatter(x=timeArr, y=filteredObs[:, 0], title='observations through time')
# # fig = px.scatter(x=timeArr, y=lObs[:, 0], title='observations through timel')
# # fig.add_scatter(x=timeArr, y=filteredObs[:, 1], name='x_dot')
# # fig.add_scatter(x=timeArr, y=lObs[:, 1], name='x_dotl')
# #theta
# # fig.add_scatter(x=timeArr, y=filteredObs[:, 4], name='theta_dot')
# # fig.add_scatter(x=timeArr, y=actArr, name='actionThroughTime')
# # fig.add_scatter(x=timeArr, y=filteredObs[:, 4], name='theta_dot')
# # fig.show()
# plt.plot(timeArr,tension,'ro')
# plt.show()
#
# regA=np.array()