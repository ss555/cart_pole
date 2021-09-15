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

dataInf = np.load('./EJPH/real-cartpole/dqn/inference_results.npz')
dataInf.allow_pickle=True
#monitor file
data,name = load_data_from_csv('./EJPH/real-cartpole/dqn/monitor.csv')
timesteps = np.zeros((16,))
rewsArr = dataInf["modelRewArr"]
obsArr = dataInf["modelsObsArr"]
actArr = dataInf["modelActArr"]
nameArr = dataInf["filenames"]
epReward = np.zeros((16,))
for i in range(0,len(obsArr)):
    print()
    obs = obsArr[i]
    act = actArr[i]
    epReward[i] = np.sum(rewsArr[i])
    timesteps[i] = np.sum(data['l'][:(i*10)])
    print(f'it {i} and {epReward[i]}')
plt.plot(timesteps,epReward)
plt.show()


