import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
sys.path.append(os.path.abspath('./../..'))
import glob
import seaborn as sns
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN, SAC
from env_custom import CartPoleButter
from custom_callbacks import EvalCustomCallback, EvalThetaDotMetric, moving_average
from matplotlib import rcParams, pyplot as plt
from custom_callbacks import plot_results
import plotly.express as px
from bokeh.palettes import d3
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from env_wrappers import load_results, ts2xy, load_data_from_csv

PLOT_TRAINING_REWARD=True
PLOT_EVAL_REWARD=True
TENSION_PLOT = True
TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
SCALE = 1.2
f_aAr = 20.75180095541654,  # -21.30359185798466,
f_bAr = 1.059719258572224,  # 1.1088617953891196,
f_cAr = 1.166390864012042*np.array([0, 0.1, 1, 10]),  # -0.902272006611719,


f_d = 0.09727843708918459,  # 0.0393516077401241, #0.0,#
wAngular = 4.881653071189049,
kPendViscousAr = 0.0706*np.array([0, 0.1, 1, 10]).T
legsStatic = np.array([np.round(f_cc,4) for f_cc in f_cAr]).T# 0.0,#
# kPendViscous = round(float(0.07035332644615992),4)
# f_c=round(float(1.166390864012042),4)
legsVisc = [round(kPendViscous,4) for kPendViscous in kPendViscousAr]
#plot params
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = 'Georgia'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["figure.dpi"] = 100
colorPalette = d3['Category20'][20]
# set labels outside
coords = [-0.15, 0.9]
fontSize = 24


NUM_TIMESTEPS = 150000
EVAL_NUM_STEPS = 5000
timesteps = np.linspace(EVAL_NUM_STEPS, NUM_TIMESTEPS, int(NUM_TIMESTEPS / EVAL_NUM_STEPS))

xl = 'Timesteps'
yl = 'Rewards'

logdir='./plots'
STEPS_TO_TRAIN=100000
EP_STEPS=800
Te=0.05
#FOLDER DIRS
dirTension = './EJPH/tension-perf'
dirStatic = './EJPH/static-friction'
dirDynamic = './EJPH/dynamic-friction'
dirNoise = './EJPH/encoder-noise'
dirAction = './EJPH/action-noise'
dirReset = './EJPH/experimental-vs-random'
#TITLES IF NEEDED
t1="Effect of applied tension on training reward"
t2='Effect of static friction on training reward'
t3='Effect of viscous friction of a pendulum on training reward'
t4='Effect of measurement noise on training reward'
t5='Effect of action noise on training reward (std in %)'
t6='Effect of initialisation on training reward'

PLOT_TRAINING_REWARD=True
PLOT_EVAL_REWARD=True
TENSION_PLOT = True
TENSION_RANGE = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
SCALE = 1.2
LABEL_SIZE = 14

# path = './EJPH/real-cartpole/dqn/inference_results.npz'
# data

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

filenames = ['./EJPH/real-cartpole/dqn_7.1V/inference_results.npz', './weights/dqn2.4V/inference_results.npz']
tensions = [7.1, 2.4]
colorId = [4,0]
legsT = [str(tension) + 'V' for tension in tensions]
SCALE=1.2
fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(SCALE*6,SCALE*2*3.7125))
for ind,file in enumerate(filenames):
    dataInf = np.load(file)
    dataInf.allow_pickle=True
    #monitor file
    data,name = load_data_from_csv('./weights/backup/dqn2.4V(copy)/monitor.csv')
    # data,name = load_data_from_csv('./EJPH/real-cartpole/dqn/monitor.csv')
    timesteps = np.zeros((16,))
    rewsArr = dataInf["modelRewArr"]
    obsArr  = dataInf["modelsObsArr"]
    actArr  = dataInf["modelActArr"]
    nameArr = dataInf["filenames"]
    epReward = np.zeros((16,))
    for i in range(0,len(obsArr)):
        print()
        epReward[i] = np.sum(rewsArr[i])
        timesteps[i] = np.sum(data['l'][:(i*10)])
        print(f'it {i} and {epReward[i]}')
    best = np.array(obsArr[np.argmax(epReward)])

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

ax[0].set_xlabel('timesteps', fontSize=LABEL_SIZE)
ax[0].set_ylabel('x [m]', fontSize=LABEL_SIZE)

ax[1].set_xlabel('timesteps', fontSize=LABEL_SIZE)
ax[1].set_ylabel('$\Theta$ [rad]', fontSize=LABEL_SIZE)
ax[0].grid()
ax[1].grid()

ax[0].text(coords[0], coords[1], chr(97) + ')', transform=ax[0].transAxes, fontsize='x-large')  # font={'size' : fontSize})
ax[1].text(coords[0], coords[1], chr(98) + ')', transform=ax[1].transAxes, fontsize='x-large')
fig.legend(legsT, loc='upper center', bbox_to_anchor=(0.5, 0.95), title="Voltage", ncol=len(legsT))
# fig.tight_layout()
fig.savefig('./EJPH/plots/theta_x.pdf')
fig.show()

