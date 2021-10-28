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
from bokeh.palettes import d3
colorPalette = d3['Category20'][20]
colorArr = [4, 0]
plt.style.use('dark_background')

def animate_wrapper(xData, yData, lineIn, dataDot, ax=None, fig=None, colorArray = [4, 0], dotMode=None):

  # fig.set_color
  # several lines in the same plot

  lineArray = []

  def animateLine(i):  #
    for k,data in enumerate(yData):

      lineIn[k].set_data(xData[0:i], data[0:i])
      if dotMode == 'everyPoint':
        dataDot[k].set_data(xData[0:i], data[0:i])
      elif dotMode == 'end':
        dataDot[k].set_data(xData[i], data[i])
      if dotMode=='everyPoint' or dotMode=='end':
        lineArray.append(dataDot[k])
      lineArray.append(lineIn[k])
    return lineArray,

  return animateLine

def animateFromData(saveVideoName, xData, yData, axisColor='k', xlabel='Timesteps', ylabel='Normalised reward',
                    videoResolution='1080x720', dotMode=False, colorArray = [4, 0],  fps=1, title='Inference reward evolution', lineColorId=None):
  '''

  :param saveVideoName:
  :param xData:
  :param yData:
  :param axisColor:
  :param xlabel:
  :param ylabel:
  :param videoResolution:
  :param dotMode: either 'everyPoint' or 'end' or None
  :param fps:
  :param title:
  :return:
  '''
  figIn,axIn = plt.subplots()
  yData = np.array(yData)
  # axIn.set_xlabel('Time [s]')

  # axIn.tick_params(labelcolor='tab:orange')
  # axIn.set_xticks([])
  # axIn.set_yticks([])
  figIn, axIn = plt.subplots()
  axIn.set_facecolor(axisColor)
  axIn.set_xlim(0, xData[-1])
  margin = 0.2
  axIn.set_ylim(yData.min() - margin, yData.max() + margin)
  axIn.set_xlabel(xlabel)
  axIn.set_ylabel(ylabel)
  axIn.set_title(title)

  dimension = np.array(np.array(yData).shape).shape
  if dimension[0]==1:
    yData = yData.reshape(-1,yData.shape[0])
  lineIn, dataDot = [], []
  for k in range(yData.shape[0]):
    line, = axIn.plot([], [], lw=2, color=colorPalette[colorArray[k]])
    Dot, = axIn.plot([], [], 'o', color=colorPalette[colorArray[k]])
    lineIn.append(line)
    dataDot.append(Dot)
  myAnimation = FuncAnimation(figIn,
                              animate_wrapper(xData, yData, lineIn = lineIn, dataDot = dataDot, ax = axIn, fig = figIn,
                                              colorArray=colorArray, dotMode = dotMode),
                              frames=np.arange(1, yData.shape[-1]), interval=100000)

  myAnimation.save(saveVideoName, fps=fps, extra_args=['-vcodec', 'libx264', '-s', videoResolution])



if __name__ == '__main__':
  Te = 0.05
  NUM_STEPS = 800
  # figure and axes containing the signals
  # fig,(axIn,ang_ax) = plt.subplots(2,1)




  # LENGTH = len(xTraining7)
  # # define the signalse
  # fig = plt.figure()
  # ax = plt.axes(xlim=(0, xTraining7[-1]), ylim = (min(yTraining7), max(yTraining7)))
  # ax.set_xlabel('Time [s]')
  # ax.set_ylabel('Normalised reward')
  # line,  = ax.plot([],[],lw=2)
  # ax.set_title('Training reward evolution')



  # axIn = plt.axes(xlim=(0, xTraining7[-1]), ylim=(min(yTraining7), max(yTraining7)))
  # axIn.set_xlabel('Time [s]')
  # axIn.set_ylabel('Normalised reward')
  # line,  = ax.plot([],[],lw=2)
  # ax.set_title('Position(m)')


  monitorFilename = ['./EJPH/real-cartpole/dqn_7.1V','./weights/dqn50-real/pwm51']
  videoTrainFilename = ['./EJPH/training_dqn7.mp4','./EJPH/training_dqn2.mp4']
  #training reward
  for video,monitor,i in zip(videoTrainFilename,monitorFilename, np.arange(2)):
    xTraining7, yTraining7, _ = plot_results(monitor, only_return_data=True)
    xTraining7 = xTraining7[0]  # *Te # time(s) frame
    yTraining7 = yTraining7[0] / NUM_STEPS  # reward per step
    # create animation using the animate() function
    animateFromData(xData=xTraining7, yData=yTraining7, colorArray = [colorArr[i]], saveVideoName=video, title='Training reward evolution')



  #inference
  Timesteps7, epRew7 = inferenceResCartpole('./EJPH/real-cartpole/dqn_7.1V/inference_results.npz',
                                            monitorFileName='./EJPH/real-cartpole/dqn_7.1V/monitor.csv')
  epRew7/=NUM_STEPS
  #inference
  animateFromData(xData=Timesteps7, yData=epRew7, dotMode='everyPoint', colorArray = [colorArr[0]], saveVideoName='./EJPH/dqn7inf.mp4')
  # 2.4 V
  Timesteps3, epRew3 = inferenceResCartpole('./weights/dqn50-real/pwm51/inference_results.npz',
                                            monitorFileName='./weights/dqn50-real/pwm51/monitor.csv')
  animateFromData(xData=Timesteps3, yData=epRew3/NUM_STEPS, dotMode='everyPoint', colorArray = [colorArr[1]], saveVideoName='./EJPH/dqn2.4inf.mp4')
  #rew best inference

  # # theta, x best inf

