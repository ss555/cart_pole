import sys
import os
sys.path.append(os.path.abspath('./')) #add subfolders in path
sys.path.append(os.path.abspath('./..'))
sys.path.append(os.path.abspath('./../..'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from custom_callbacks import plot_results
from utils import inferenceResCartpole


def animate_wrapper(xData, yData, dataDot=None, ax=None, fig=None, color=None):

  # fig.set_color
  # several lines in the same plot

  artists = []
  lineArray = []
  for data, k in enumerate(yData):
    lineIn, = ax.plot([], [], lw=2)
    lineArray.append(lineIn)
  def animateLine(i):  # ,xlabel,ylablel,title):
    try:
      lineIn.set_data(xData[0:i], yData[0:i,k])
      if dataDot is not None:
        dataDot.set_data(xData[0:i], yData[0:i,k])
        return lineIn, dataDot
      return lineIn,
    except Exception as e:
      print(e)

  return animateLine

def animateFromData(saveVideoName, xData, yData, axisColor='k', xlabel='Timesteps', ylabel='Reward/step',
                    videoResolution='1080x720', dotAtEveryPoint=False):
  figIn,axIn = plt.subplots()
  yData = np.array(yData)
  # axIn.set_xlabel('Time [s]')
  axIn.set_ylabel(ylabel)
  # axIn.tick_params(labelcolor='tab:orange')
  # axIn.set_xticks([])
  # axIn.set_yticks([])
  figIn, axIn = plt.subplots()
  axIn.set_facecolor(axisColor)
  axIn.set_xlim(0, xData[-1])
  margin = 0.2
  axIn.set_ylim(yData.min() - margin, yData.max() + margin)
  axIn.set_xlabel(xlabel)
  axIn.set_title('Inference reward evolution')
  if dotAtEveryPoint:
    dataDot, = axIn.plot([], [], 'o')
  else:
    dataDot = None

  myAnimation = FuncAnimation(figIn,
                              animate_wrapper(xData, yData, dataDot, axIn, figIn, color='#eafff5'),
                              frames=np.arange(1, yData.shape[-1]), interval=100000)

  myAnimation.save(saveVideoName, fps=1, extra_args=['-vcodec', 'libx264', '-s', videoResolution])

class Signal(object):
  def __init__(self, name, color, xData, yData, pos_ax, ang_ax):
    self.data = data
    self.tpast = tpast
    self.tfuture = tfuture
    self.dt = (self.t[-1]-self.t[0])/(len(self.t)-1)
    self.pos = data.state(0)
    self.ang = data.state(1)
    self.pos_line, = pos_ax.plot(self.t, self.pos, '-', label=name, color=color)
    self.pos_pred_line, = pos_ax.plot([], [], '--', color=color)
    self.pos_dot, = pos_ax.plot([], [], 'o', color=color)
    self.ang_line, = ang_ax.plot(self.t, self.pos, '-', label=name, color=color)
    self.ang_pred_line, = ang_ax.plot([], [], '--', color=color)
    self.ang_dot, = ang_ax.plot([], [], 'o', color=color)
    self.pos_ax = pos_ax
    self.ang_ax = ang_ax
    self.idot = int(self.tpast/self.dt)

  def update(self, i):
    # i is the current iteration
    # istart represents the beginning of the signal portion to be shown
    istart = max(0, i-self.idot)
    # show the past and current positon/angle
    self.pos_line.set_data(self.t[istart:i+1], self.pos[istart:i+1])
    self.pos_dot.set_data([self.t[i]], [self.pos[i]])
    self.ang_line.set_data(self.t[istart:i+1], self.ang[istart:i+1])
    self.ang_dot.set_data([self.t[i]], [self.ang[i]])
    return [
      self.pos_line, self.pos_pred_line, self.pos_dot,
      self.ang_line, self.ang_pred_line, self.ang_dot
    ]

if __name__ == '__main__':
  Te = 0.05
  NUM_STEPS = 800
  # figure and axes containing the signals
  # fig,(axIn,ang_ax) = plt.subplots(2,1)


  #import from csv
  xArrEx, yArrEx, _ = plot_results('./EJPH/real-cartpole/dqn_7.1V', only_return_data=True)
  xArrEx = xArrEx[0]#*Te # time(s) frame
  yArrEx = yArrEx[0]/NUM_STEPS # reward per step

  # LENGTH = len(xArrEx)
  # # define the signalse
  # fig = plt.figure()
  # ax = plt.axes(xlim=(0, xArrEx[-1]), ylim = (min(yArrEx), max(yArrEx)))
  # ax.set_xlabel('Time [s]')
  # ax.set_ylabel('Reward/step')
  # line,  = ax.plot([],[],lw=2)
  # ax.set_title('Training reward evolution')



  # axIn = plt.axes(xlim=(0, xArrEx[-1]), ylim=(min(yArrEx), max(yArrEx)))
  # axIn.set_xlabel('Time [s]')
  # axIn.set_ylabel('Reward/step')
  # line,  = ax.plot([],[],lw=2)
  # ax.set_title('Position(m)')



  # def animate(i):#,xlabel,ylablel,title):
  #   try:
  #     line.set_data(xArrEx[0:i],yArrEx[0:i])
  #     return line,
  #   except Exception as e:
  #     print(e)





  # create animation using the animate() function
  # myAnimation = FuncAnimation(fig, animate, frames=np.arange(1,LENGTH), interval=100000)
  # myAnimation.save('./EJPH/training_dqn7.mp4', fps=30, extra_args=['-vcodec', 'libx264','-s', '1080x720'])

  #inference
  Timesteps7, epRew7 = inferenceResCartpole('./EJPH/real-cartpole/dqn_7.1V/inference_results.npz',
                                            monitorFileName='./EJPH/real-cartpole/dqn_7.1V/monitor.csv')
  epRew7/=NUM_STEPS


  # axIn = plt.axes(xlim=(0, Timesteps7[-1]), ylim=(min(epRew7), max(epRew7)))
  # axIn.set_xlabel('Time [s]')
  # axIn.set_ylabel('Reward/step')
  # # axIn.set_xticks([])
  # # axIn.set_yticks([])
  # axIn.set_facecolor('k')
  # # pos_fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
  # lineIn, = axIn.plot([],[],lw=2)
  # dataDot, = axIn.plot([],[],'o')
  # axIn.set_title('Inference reward evolution')



  #inference
  animateFromData(xData=Timesteps7,yData=epRew7,saveVideoName='./EJPH/dqn7inf.mp4')
  # 2.4 V
  Timesteps3, epRew3 = inferenceResCartpole('./weights/dqn50-real/pwm51/inference_results.npz',
                                            monitorFileName='./weights/dqn50-real/pwm51/monitor.csv')
  animateFromData(xData=Timesteps3, yData=epRew3/NUM_STEPS, saveVideoName='./EJPH/dqn2.4inf.mp4')
  #rew best inference

  # # theta, x best inf

