import sys
import os
import numpy as np
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
sys.path.append(os.path.abspath('./../..'))
from matplotlib import rcParams, pyplot as plt
from custom_callbacks import plot_results
import plotly.express as px
from bokeh.palettes import d3
from env_wrappers import load_results, ts2xy, load_data_from_csv
from generate_video_with_caption import animateFromData
PLOT_TRAINING_REWARD = True
PLOT_EVAL_REWARD = True
TENSION_PLOT = True
ANIMATE = False

TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
SCALE = 1.2
#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 100
colorPalette = d3['Category20'][20]
# set labels outside
coords = [-0.15, 0.9]
fontsize = 24
logdir='./plots'
STEPS_TO_TRAIN=100000
EP_STEPS = 800
Te=0.05
LABEL_SIZE = 14


def calculate_angle(prev_value, cos, sin, count=0):
    '''
    :param prev_value:
    :param cos: cosinus
    :param sin: sinus
    :return:
    '''
    if prev_value - np.arctan2(sin, cos) > np.pi:
        count += 1
        return np.arctan2(sin, cos), count
    elif np.arctan2(sin, cos) - prev_value > np.pi:
        count -= 1
        return np.arctan2(sin, cos), count
    return np.arctan2(sin, cos), count

filenamesNpz = ['./weights/dqn50-real/pwm151/inference_results.npz', './weights/dqn50-real/pwm51/inference_results.npz']
filenamesCsv = ['./weights/dqn50-real/pwm151/monitor.csv', './weights/dqn50-real/pwm51/monitor.csv']
#old
# filenamesNpz = ['./EJPH/real-cartpole/dqn_7.1V/inference_results.npz', './weights/dqn50-real/pwm51/inference_results.npz']
# filenamesCsv = ['./EJPH/real-cartpole/dqn_7.1V/monitor.csv', './weights/dqn50-real/pwm51/monitor.csv']
# filenames = ['./EJPH/real-cartpole/dqn_7.1V/inference_results.npz', './weights/dqn2.4V/inference_results.npz']
tensions = [7.1, 2.4]
colorId = [4,0]
legsT = [str(tension) + 'V' for tension in tensions]
SCALE=1.2
fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(SCALE*6,SCALE*2*3.7125))
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
    best = np.array(obsArr[np.argmax(epReward)])#observations for best learnt policy

    thetaArr = []
    prev_angle_value = np.arctan2(best[0,3],best[0,2])
    print(f'best: {np.argmax(epReward)}, with initial angle {prev_angle_value}')
    count_tours=0
    for j in range(best.shape[0]):
        angle, count_tours = calculate_angle(prev_angle_value, best[j,2], best[j,3], count_tours)
        prev_angle_value = angle
        thetaArr.append(angle + count_tours * np.pi * 2)
    #plot
    ax[0].plot(best[:, 0], color=colorPalette[colorId[ind]])
    ax[1].plot(thetaArr, color=colorPalette[colorId[ind]])
    thetaAnimation.append(thetaArr)
    xAnimation.append(best[:, 0])


if ANIMATE:
    animateFromData(saveVideoName='./EJPH/thetaEvolution.mp4', dotMode='end', xData=np.arange(800),yData=thetaAnimation, title='$\Theta$ evolution', ylabel='$\Theta$ [rad]', fps=50)
    animateFromData(saveVideoName='./EJPH/xEvolution.mp4', dotMode='end', xData=np.arange(800),yData=xAnimation, title='$x$ evolution', ylabel='$x$ [m]', fps=50)


ax[0].set_xlabel('Time step', fontsize=LABEL_SIZE)
ax[0].set_ylabel('x [m]', fontsize=LABEL_SIZE)
ax[1].set_xlabel('Time step', fontsize=LABEL_SIZE)
ax[1].set_ylabel('$\Theta$ [rad]', fontsize=LABEL_SIZE)
ax[0].grid()
ax[1].grid()

ax[0].text(coords[0], coords[1], chr(97) + ')', transform=ax[0].transAxes, fontsize='x-large')  # font={'size' : fontsize})
ax[1].text(coords[0], coords[1], chr(98) + ')', transform=ax[1].transAxes, fontsize='x-large')
fig.legend(legsT, loc='upper center', bbox_to_anchor=(0.5, 0.95), title="Applied tension", ncol=len(legsT))
# fig.tight_layout()
fig.savefig('./EJPH/plots/theta_x.pdf')
fig.show()

