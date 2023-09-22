import sys
import os
import numpy as np
import time
import subprocess
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule, plot
from custom_callbacks import plot_results
from env_wrappers import Monitor
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, DDQN, SAC, PPO
from env_custom import CartPoleRK4
from utils import read_hyperparameters
from pathlib import Path
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from custom_callbacks import EvalCustomCallback, EvalThetaDotMetric, moving_average
from matplotlib import rcParams, pyplot as plt
import plotly.express as px
from bokeh.palettes import d3
from distutils.dir_util import copy_tree
from env_wrappers import VideoRecorderWrapper
from glob import glob
import pandas as pd
from rlutils.utils import config_paper,moving_average,load_results

#TODO use subprocess to parallelise sim
c=config_paper()
STEPS_TO_TRAIN = 150000
EP_STEPS = 800
Te = 0.05
MANUAL_SEED = 1
video_folder = None
# simulation results
tr_plot=False
inf_plot=True
logdir='/media/sardor/b/12-STABLE3/EJPH/comapre/'
if tr_plot:
    '''plot moving average of monitor.csv'''
    filedirs=os.listdir(logdir)
    fig,ax=plt.subplots()
    for filedir in filedirs:
        if 'discr_sim' in filedir:
            print(filedir)
            filedir=os.path.join(logdir,filedir)
            filedir=os.path.join(filedir,'monitor.csv')
            try:
                df = pd.read_csv(filedir,index_col=None) #data_frame, legends = load_results(log_folder)
                print(df.head())
            except:
                print(f'error occured treating folder : {filedir}')
                sys.exit(0)
            ax.plot(moving_average(df['r'],window=100),label=filedir.split('/')[-2].split('_')[-1])

    ax.legend()
    plt.savefig(os.path.join(logdir,'discr_sim_tr.png'))
    plt.show()

if inf_plot:
    '''plot moving average of inference'''
    filedirs=os.listdir(logdir)
    fig,ax=plt.subplots()
    # fig1,ax1=plt.subplots()
    # fig2,ax2=plt.subplots()
    for filedir in filedirs:
        if 'cont_sim' in filedir:
        # if 'discr_sim' in filedir:
            print(filedir)
            filedir=os.path.join(logdir,filedir)
            filedir=os.path.join(filedir,'*.npz')
            filedir=glob(filedir)
            print(filedir)
            for file in filedir:
                print(file)
                data=np.load(file)
                ax.plot(data['timesteps'],np.mean(data['results'],axis=-1)/800,label=file.split('/')[-2].split('_')[-1])
    ax.legend()
    # ax.set_title('inference reward')
    ax.set_ylim(-0.4,1)
    ax.set_xlabel('time steps')
    ax.set_ylabel('inference reward')
    plt.savefig(os.path.join(logdir,f'inf_sim_inf{filedir}.pdf'))
    plt.show()