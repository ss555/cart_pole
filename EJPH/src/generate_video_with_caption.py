import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

if __name__ == '__main__':

  # figure and axes containing the signals
  # fig,(pos_ax,ang_ax) = plt.subplots(2,1)
  pos_fig,pos_ax = plt.subplots()
  ang_fig,ang_ax = plt.subplots()

  #import from csv

  # define the signals
  ref_time = data.get('simple').custom_data().get('reference_signal').get('time')
  ref_position = data.get('simple').custom_data().get('reference_signal').get('position')
  time = data.get('simple').time(time_shift=True)
  # positions = { test : data.get(test).state(0) for test in tests }

  # figure and axes containing the signals
  # fig,(pos_ax,ang_ax) = plt.subplots(2,1)
  pos_fig,pos_ax = plt.subplots()
  ang_fig,ang_ax = plt.subplots()

  pos_ax.set_xticks([])
  pos_ax.set_yticks([])
  pos_ax.set_facecolor('k')
  pos_fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
  ang_ax.set_xticks([])
  ang_ax.set_yticks([])
  ang_ax.set_facecolor('k')
  ang_fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

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
  ang_ax.plot(ref_time, [np.pi]*len(ref_time), '-w', label='reference')
  max_angle = 0.15
  ang_ax.set_ylim(np.pi-max_angle, np.pi+max_angle)

  # generate the lines
  colors = {
    "simple": "#BF0000",
    "lerp": "#0000FF",
    "laguerre": "#800080"
  }
  signals = [
    Signal(
      test,
      colors.get(test),
      data.get(test),
      tpast,
      tfuture,
      pos_ax,
      ang_ax,
      with_prediction = with_prediction
    )
    for test in tests
  ]

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
    ang_ax.set_xlim(time[istart], time[istop])
    return artists

  # pos_ax.legend()
  # ang_ax.legend()
  print("Creating animations")
  max_frame = int(tmax/dt)-2
  pos_animation = FuncAnimation(pos_fig, update, max_frame, interval=int(1000*dt))
  ang_animation = FuncAnimation(ang_fig, update, max_frame, interval=int(1000*dt))

  print("Saving animations")
  fps = 1/dt
  animation_pairs = [
    (pos_animation, "position"),
    (ang_animation, "angle")
  ]
  for animation,animation_name in animation_pairs:
    out_name = prefix + animation_name + ("-with-prediction" if with_prediction else "") + ".mp4"
    print("Saving", out_name)
    animation.save(out_name, writer=FFMpegWriter(fps=fps))
  # pos_animation.save(prefix + "position.mp4", writer=FFMpegWriter(fps=fps))
  # ang_animation.save(prefix + "angle.mp4", writer=FFMpegWriter(fps=fps))

  plt.show()