import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from custom_callbacks import plot_results

if __name__ == '__main__':
  Te = 0.05
  # figure and axes containing the signals
  # fig,(pos_ax,ang_ax) = plt.subplots(2,1)


  #import from csv
  xArrEx, yArrEx, _ = plot_results('./EJPH/real-cartpole/dqn_7.1V', only_return_data=True)
  xArrEx = xArrEx[0]*Te # time(s) frame
  yArrEx = yArrEx[0]/800 # reward per step

  LENGTH = len(xArrEx)
  # define the signals
  fig = plt.figure()
  ax = plt.axes(xlim=(0, xArrEx[-1]), ylim = (min(yArrEx), max(yArrEx)))
  ax.set_xlabel('Time [s]')
  ax.set_ylabel('Reward/step')
  line,  = ax.plot([],[],lw=2)
  # fig,ax = plt.subplots()
  ax.set_title('Training reward evolution')

  def animate(i,xArrEx,yArrEx):
    # curve.set_data(xArrEx[0:i],yArrEx[0:i]*0.05)
    try:
      line.set_data(xArrEx[0:i],yArrEx[0:i])
      return line,
    except Exception as e:
      print(e)

  # pos_ax.set_xticks([])
  # pos_ax.set_yticks([])
  # pos_ax.set_facecolor('k')
  # pos_fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

  # create animation using the animate() function
  myAnimation = FuncAnimation(fig, animate(xArrEx,yArrEx), frames=np.arange(1,LENGTH), interval=100000)
  myAnimation.save('./EJPH/training_dqn7.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

  myAnimation = FuncAnimation(fig, animate(xArrEx,yArrEx), frames=np.arange(1,LENGTH), interval=100000)
  myAnimation.save('./EJPH/training_dqn7.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

  fig.show()
  '''
  # settings for plotting
  with_prediction = False
  dt = (time[-1]-time[0])/(len(time)-1)
  tmin = 0
  if with_prediction:
    tpast = 2.5
    tfuture = 1.5
  else:
    tpast = 3.0
    tfuture = 1.0
  tmax = time[-1] - tfuture

  # plot the reference
  pos_ax.plot(ref_time, ref_position, '-w', label='reference')

  # generate the lines
  colors = {
    "simple": "#BF0000"
  }

  # to be called in the beginning of the animation
  def init():
    return update(0)

  # to be called to update the animation
  def update(i):
    # draw the first part of the signals
    artists = []
    for signal in signals:
      artists += signal.update(i)
    # set the limits of the axes
    istart = max(0, i-int(tpast/dt))
    istop = istart + int((tpast+tfuture)/dt)
    pos_ax.set_xlim(time[istart], time[istop])
    return artists

  # pos_ax.legend()
  # ang_ax.legend()
  print("Creating animations")
  max_frame = int(tmax/dt)-2
  pos_animation = FuncAnimation(pos_fig, update, max_frame, interval=int(LENGTH*dt))

  print("Saving animations")
  fps = 1/dt
  animation_pairs = [
    (pos_animation, "training"),
  ]
  for animation,animation_name in animation_pairs:
    out_name = animation_name + ("-reward" if with_prediction else "") + ".mp4"
    print("Saving", out_name)
    animation.save(out_name, writer=FFMpegWriter(fps=fps))
  pos_animation.save("position.mp4", writer=FFMpegWriter(fps=fps))

  plt.show()
  '''