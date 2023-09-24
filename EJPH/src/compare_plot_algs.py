import sys
import os
import numpy as np
import time
import subprocess
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from src.utils import linear_schedule, plot
from src.custom_callbacks import plot_results
from src.env_wrappers import Monitor
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, DDQN, SAC, PPO
from src.env_custom import CartPoleRK4
from src.utils import read_hyperparameters
from pathlib import Path
from src.custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from src.custom_callbacks import EvalCustomCallback, EvalThetaDotMetric, moving_average
from matplotlib import rcParams, pyplot as plt
import plotly.express as px
from bokeh.palettes import d3
from distutils.dir_util import copy_tree
from src.env_wrappers import VideoRecorderWrapper
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
logdir='./EJPH/comapre/'
fig, ax = plt.subplots(1,2,figsize=(14,5))
if tr_plot:
    '''plot moving average of monitor.csv'''
    filedirs=os.listdir(logdir)
    # fig,ax=plt.subplots()
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
            ax[0].plot(moving_average(df['r'],window=100),label=filedir.split('/')[-2].split('_')[-1])

    ax[0].legend()
    # plt.savefig(os.path.join(logdir,'discr_sim_tr.png'))
    # plt.show()

if inf_plot:
    '''plot moving average of inference'''
    filedirs=os.listdir(logdir)
    # fig1,ax1=plt.subplots()
    # fig2,ax2=plt.subplots()
    for filedir in filedirs:
        for a,inf in zip(ax,['discr_sim','cont_sim']):
            if inf in filedir:
                print(filedir)
                filedir=os.path.join(logdir,filedir)
                filedir=os.path.join(filedir,'*.npz')
                filedir=glob(filedir)
                print(filedir)
                for file in filedir:
                    print(file)
                    data=np.load(file)
                    a.plot(data['timesteps'],np.mean(data['results'],axis=-1)/800,label=file.split('/')[-2].split('_')[2])
    for a in ax:
        a.legend()
        # a.set_title('inference reward')
        a.set_ylim(-0.4,1)
        a.set_xlabel('time steps')
        a.set_ylabel('inference reward')

    ax[0].text(-0.1, 1.05, 'a)', transform=ax[0].transAxes, fontweight='bold', va='top', ha='right')
    ax[1].text(-0.1, 1.05, 'b)', transform=ax[1].transAxes, fontweight='bold', va='top', ha='right')

    plt.savefig(os.path.join(logdir,f'inf_sim_inf_cont.pdf'))
    plt.show()