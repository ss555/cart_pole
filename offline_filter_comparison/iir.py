import numpy as np
import pylab as pl
import scipy as sp
import iir_filter
from scipy import signal
from scipy.fft import fft, fftfreq
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath(os.getcwd()+'/offline_filter_comparison'))
#filtering of identification data to understand cutoff frequency

#of data
for item in sys.path:
    print(item)
obsArr = np.array(pd.read_csv(os.getcwd()+"/offline_filter_comparison/obs.csv")).reshape(-1,5)
timeArr=np.array(pd.read_csv(os.getcwd()+"/offline_filter_comparison/time.csv")).reshape(-1,)
actArr =np.array(pd.read_csv(os.getcwd()+"/offline_filter_comparison/actArr.csv")).reshape(-1,)
obsArr=obsArr-1
x_dot=obsArr[:,1]
theta_dot=obsArr[:,4]
dt=np.mean(np.diff(timeArr))
yf=fft(x_dot)
xf=fftfreq(len(x_dot),0.05)
fc=0.3
# sos = signal.butter(4, fc, 'lowpass', output='sos')
fX_dot = iir_filter.IIR_filter(signal.butter(4, fc, 'lowpass', output='sos'))
fTheta_dot = iir_filter.IIR_filter(signal.butter(4, fc, 'lowpass', output='sos'))
x_dotF=np.zeros_like(x_dot)
theta_dotF=np.zeros_like(theta_dot)
for i in range(len(x_dot)):
    x_dotF[i]=fX_dot.filter(x_dot[i])
    theta_dotF[i]=fTheta_dot.filter(theta_dot[i])
yf2=fft(x_dotF)
fig = make_subplots(rows=3, cols=1)
fig.add_trace(go.Scatter(x=timeArr,y=x_dot, mode="lines", name='x_dot'))#,label='Xd')
fig.add_trace(go.Scatter(x=timeArr,y=x_dotF, name='x_dot_fourrier'), row=1, col=1)#,label='XdF')
fig.add_trace(go.Scatter(x=timeArr,y=theta_dot, name='theta_dot'), row=3, col=1)#,label='Xd')
fig.add_trace(go.Scatter(x=timeArr,y=theta_dotF, name='theta_dotF'), row=3, col=1)#,label='XdF')
fig.add_trace(go.Scatter(x=xf,y=abs(yf)), row=2, col=1, name='fft_of_x_dot')#,label='fft')
fig.add_trace(go.Scatter(x=xf,y=abs(yf2)), row=2, col=1, name='fft_of_x_dotF')#,label='fft2')

fig.show()
#EXAMPLE
# data = np.loadtxt(os.getcwd()+'/offline_filter_comparison/ecg_50hz_1.dat')
# data = data - 2048
# data = data * 2E-3 * 500 / 1000
# fs = 1000.0
# pl.title('ECG')
# #
# y = data[:,1]
# t = data[:,0]
# pl.subplot(311)
# pl.plot(t,y)
# pl.xlabel('time/sec')
# pl.ylabel('ECG/mV')
# #
# f0 = 48.0
# f1 = 52.0
# sos = signal.butter(2, [f0/fs*2,f1/fs*2], 'bandstop', output='sos')
#
# f = iir_filter.IIR_filter(sos)
# y2 = np.zeros(len(y))
# for i in range(len(y)):
#     y2[i] = f.filter(y[i])
# yf = fft(y2)
# yf[0] = 0
#
#
# fig = make_subplots(rows=3, cols=1)
# fig.add_trace(go.Scatter(x=np.linspace(0,fs,len(yf)),y=abs(yf), mode="lines",text='fft'))
# fig.show()#, row=1, col=1)
# fig.add_trace(go.Scatter(x=t,y=y), row=2, col=1)
# fig.add_trace(go.Scatter(x=t,y=y2), row=2, col=1)
#
# fig.show()