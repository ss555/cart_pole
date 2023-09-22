import sys
import os
sys.path.append(os.path.abspath('/')) #add subfolders in path
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../../tmp'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from src.custom_callbacks import plot_results
from src.utils import inferenceResCartpole
from bokeh.palettes import d3

colorPalette = d3['Category20'][20]
colorArr = [4, 0]
plt.style.use('dark_background')
Te = 0.05
NUM_STEPS = 800
labelFont = 14

monitorFilename = ['./weights/dqn50-real/pwm151', './weights/dqn50-real/pwm51']
# monitorFilename = ['./deepRl/real-cartpole/dqn_7.1V', './weights/dqn50-real/pwm51']
videoTrainFilename = ['./deepRl/training_dqn7.mp4', './deepRl/training_dqn2.mp4']
inferenceName = ['./weights/dqn50-real/pwm151/inference_results.npz', './weights/dqn50-real/pwm51/inference_results.npz']
# inferenceName = ['./deepRl/real-cartpole/dqn_7.1V/inference_results.npz','./weights/dqn50-real/pwm51/inference_results.npz']
videoInfFilename = ['./deepRl/dqn7inf.mp4', './deepRl/dqn2.4inf.mp4']

def animate_wrapper(xData, yData, lineIn, dataDot, ax=None, fig=None, colorArray = [4, 0], dotMode=None):
  # several lines in the same plot
  lineArray = []
  def animateLine(i):  #
    for k, data in enumerate(yData):
      lineIn[k].set_data(xData[0:i], data[0:i])
      if dotMode == 'everyPoint':
        dataDot[k].set_data(xData[0:i], data[0:i])
      elif dotMode == 'end':
        dataDot[k].set_data(xData[i], data[i])
      if dotMode == 'everyPoint' or dotMode == 'end':
        lineArray.append(dataDot[k])
      lineArray.append(lineIn[k])
    return lineArray,
  return animateLine

def animateFromData(saveVideoName, xData, yData, axisColor='k', xlabel='Time step', ylabel='Normalized reward',
                    videoResolution='1080x720', dotMode=False, colorArray = [4, 0],  fps=1, title='Inference reward evolution'):
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
  figIn, axIn = plt.subplots()
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
  axIn.set_xlabel(xlabel, fontsize=labelFont)
  axIn.set_ylabel(ylabel, fontsize=labelFont)
  axIn.set_title(title, fontsize=labelFont)

  dimension = np.array(np.array(yData).shape).shape
  if dimension[0]==1:
    yData = yData.reshape(-1, yData.shape[0])


  lineIn, dataDot = [], []
  for k in range(yData.shape[0]):
    line, = axIn.plot([], [], lw=2, color=colorPalette[colorArray[k]])
    Dot, = axIn.plot([], [], 'o', color=colorPalette[colorArray[k]])
    lineIn.append(line)
    dataDot.append(Dot)
  myAnimation = FuncAnimation(figIn,
                              animate_wrapper(xData, yData, lineIn = lineIn, dataDot = dataDot, ax = axIn, fig = figIn,
                                              colorArray=colorArray, dotMode = dotMode),
                              frames=np.arange(1, yData.shape[-1]+1), interval=100000)

  myAnimation.save(saveVideoName, fps=fps, extra_args=['-vcodec', 'libx264', '-s', videoResolution])



if __name__ == '__main__':
  TRAINING_VIDEO = False
  if TRAINING_VIDEO:
    #training reward
    for video,monitor,i in zip(videoTrainFilename,monitorFilename, np.arange(2)):
      xTraining7, yTraining7, _ = plot_results(monitor, only_return_data=True)
      xTraining7 = xTraining7[0]  # *Te # time(s) frame
      yTraining7 = yTraining7[0] / NUM_STEPS  # reward per step
      # create animation using the animate() function
      animateFromData(xData=xTraining7, yData=yTraining7, colorArray = [colorArr[i]], saveVideoName=video, title='Training reward evolution')



  #inference
  Timesteps7, epRew7 = inferenceResCartpole(inferenceName[0], monitorFileName=monitorFilename[0]+'/monitor.csv')
  epRew7/=NUM_STEPS
  #inference
  animateFromData(xData=Timesteps7, yData=epRew7, dotMode='everyPoint', colorArray = [colorArr[0]], saveVideoName = videoInfFilename[0])
  # 2.4 V
  Timesteps3, epRew3 = inferenceResCartpole(inferenceName[1],
                                            monitorFileName=monitorFilename[1]+'/monitor.csv')
  epRew3 /= NUM_STEPS
  animateFromData(xData=Timesteps3, yData=epRew3, dotMode='everyPoint', colorArray = [colorArr[1]], saveVideoName = videoInfFilename[1])
  #rew best inference

  # # theta, x best inf

