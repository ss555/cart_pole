import sys
import os
import numpy as np
sys.path.append(os.path.abspath('/'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../../tmp'))
from matplotlib import rcParams, pyplot as plt
from custom_callbacks import plot_results
from src.env_wrappers import load_results, ts2xy, load_data_from_csv
from utils import calculate_angle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset,inset_axes
#plot params
fontsize=20
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = fontsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 300
colors=['green','blue']
PLOT_TRAINING_REWARD = True
PLOT_EVAL_REWARD = True
TENSION_PLOT = True
ANIMATE = False
cmap = plt.cm.brg
TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]

# set labels outside
coords = [0.02, 0.9]
fontsize = 24
logdir='./plots'
STEPS_TO_TRAIN=100000
EP_STEPS = 800
Te=0.05
LABEL_SIZE = 20


# filenamesNpz = ['./weights/dqn50-real/pwm151/inference_results.npz', './weights/dqn50-real/pwm51/inference_results.npz']
# filenamesCsv = ['./weights/dqn50-real/pwm151/monitor.csv', './weights/dqn50-real/pwm51/monitor.csv']
#old
filenamesNpz = ['./deepRl/real-cartpole/dqn_7.1V/inference_results.npz', './weights/dqn50-real/pwm51/inference_results.npz']
filenamesCsv = ['./deepRl/real-cartpole/dqn_7.1V/monitor.csv', './weights/dqn50-real/pwm51/monitor.csv']
# filenames = ['./deepRl/real-cartpole/dqn_7.1V/inference_results.npz', './weights/dqn2.4V/inference_results.npz']
tensions = [7.1, 2.4]
colorId = [4,0]
legsT = [str(tension) + 'V' for tension in tensions]
SCALE=2

fig = plt.figure(figsize=(SCALE*7.5,SCALE*2.1*3.7125))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(313)
thetaAnimation=[]
xAnimation=[]

for file, monitor, ind in zip(filenamesNpz,filenamesCsv, np.arange(2)):
    dataInf = np.load(file)
    dataInf.allow_pickle=True
    data, name = load_data_from_csv(monitor)

    rewsArr = dataInf["modelRewArr"]
    obsArr  = dataInf["modelsObsArr"]
    actArr  = dataInf["modelActArr"]
    nameArr = dataInf["filenames"]

    timesteps = np.zeros((len(obsArr),))
    epReward = np.zeros((len(obsArr),))
    for i in range(0,len(obsArr)):
        print()
        epReward[i] = np.sum(rewsArr[i])
        timesteps[i] = np.sum(data['l'][:(i*10)])
        print(f'it {i} and {epReward[i]}')
    bestObs = np.array(obsArr[np.argmax(epReward)])#observations for bestObs learnt policy
    actArrObs = np.array(actArr[np.argmax(epReward)])#observations for bestObs learnt policy

    thetaArr = []
    prev_angle_value = np.arctan2(bestObs[0,3],bestObs[0,2])
    print(f'bestObs: {np.argmax(epReward)}, with initial angle {prev_angle_value}')
    count_tours=0
    for j in range(bestObs.shape[0]):
        angle, count_tours = calculate_angle(prev_angle_value, bestObs[j,2], bestObs[j,3], count_tours)
        prev_angle_value = angle
        thetaArr.append(angle + count_tours * np.pi * 2)
    #plot
    ax1.plot(bestObs[:, 0], color=colors[ind])
    ax3.plot(thetaArr, color=colors[ind])
    ax2.plot(bestObs[:,0], bestObs[:,1], color=colors[ind])
    if ind==0:#7.1V
        ax5.plot((actArrObs-1)[:200]*7.1, color=colors[ind])
        ax4.plot(thetaArr, bestObs[:,-1], color=colors[ind])
        plt.rcParams['font.size'] = 14
        # axins = inset_axes(ax5, width="60%", height="80%", loc='lower right', borderpad=2)  # ,bbox_to_anchor=())
        # axins.plot((actArrObs-1)*7.1, color=colors[ind])
        # axins.set_xlim(700, 750)
        # mark_inset(ax5, axins, loc1=3, loc2=4, visible=True, edgecolor='green',lw=3)
        plt.rcParams['font.size'] = fontsize
    else:
        ax4.plot(thetaArr, bestObs[:, -1], linewidth=0.6, color=colors[ind])


ax1.set_xlabel('Time step', fontsize=LABEL_SIZE)
ax1.set_ylabel('x [m]', fontsize=LABEL_SIZE)
ax2.set_xlabel('x [m]', fontsize=LABEL_SIZE)
ax2.set_ylabel('$\dot{x}$ [m/s]', fontsize=LABEL_SIZE)
ax3.set_xlabel('Time step', fontsize=LABEL_SIZE)
ax3.set_ylabel('$\Theta$ [rad]', fontsize=LABEL_SIZE)

ax4.set_ylabel('$\dot{\Theta}$ [rad/s]', fontsize=LABEL_SIZE)
ax4.set_xlabel('$\Theta$ [rad]', fontsize=LABEL_SIZE)

ax5.set_xlabel('Time step', fontsize=LABEL_SIZE)
ax5.set_ylabel('Applied voltage $[V]$', fontsize=LABEL_SIZE)
#arrow
arrow_style = {
    "head_width": 10,
    "head_length": 0.2,
    "color":"k"
}

# plt.arrow(x=0, y=0, dx=110, dy=0, **arrow_style)
ax5.arrow(0, 0, 105, 0, width=0.01,head_width=0.5, head_length=5,zorder=10, fc='k', ec='k')
ax5.annotate('swing-up',
ha = 'center', va = 'center',
xytext = (40, 1), xy = (105, 0))

# shrink the y axis to fit legend
# figT.tight_layout()
# shrink = 0.07
# for tax,bax in zip(a[0],a[1]):
#     tbox = tax.get_position()
#     bbox = bax.get_position()
#     tax.set_position([tbox.x0, tbox.y0-shrink*bbox.height, tbox.width, (1-shrink)*tbox.height])
#     bax.set_position([bbox.x0, bbox.y0, bbox.width, (1-shrink)*bbox.height])

def label_ax(axis,i=0):
    axis.text(coords[0], coords[1], chr(97+i) + ')', transform=axis.transAxes,fontsize='x-large')

for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    label_ax(ax,i)
# ax1.text(coords[0], coords[1], chr(97) + ')', transform=ax1.transAxes, fontsize='x-large')  # font={'size' : fontsize})
# ax2.text(coords[0], coords[1], chr(98) + ')', transform=ax2.transAxes, fontsize='x-large')
# ax3.text(coords[0], coords[1], chr(99) + ')', transform=ax3.transAxes, fontsize='x-large')  # font={'size' : fontsize})
# ax4.text(coords[0], coords[1], chr(100) + ')', transform=ax4.transAxes, fontsize='x-large')
# ax5.text(coords[0], coords[1], chr(101) + ')', transform=ax5.transAxes, fontsize='x-large')
# fig.legend(legsT, loc='upper center', bbox_to_anchor=(0.5, 0.96), title="Applied tension", ncol=len(legsT))
# fig.tight_layout()
fig.savefig('./deepRl/plots/theta_x.pdf')
fig.show()

