import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule
from custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn import MlpPolicy
# from stable_baselines3.sac import MlpPolicy
from stable_baselines3 import DQN,SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from env_custom import CartPoleButter
# from utils import plot_results
import argparse
import yaml
import glob
import seaborn as sns
PLOT_TRAINING_REWARD=False
PLOT_EVAL_REWARD=True

def save_show_fig(xArr,yArr,legs,title,savename):
    for i in range(len(xArr)):
        sns.lineplot(y=yArr[i], x=xArr[i])
    plt.title(title)
    plt.xlabel('timesteps')
    plt.ylabel('Rewards')
    plt.legend(legs, loc='best')
    plt.savefig(savename)
    plt.show()

def generate_legends(legends):
    legends = np.array([legend.split('_') for legend in legends])
    return legends


logdir='./plots'
STEPS_TO_TRAIN=100000
EP_STEPS=800
Te=0.05
t1="Effect of applied tension on training reward"
t2='Effect of static friction on training reward'
t3='Effect of viscous friction of a pendulum on training reward'
t4='Effect of measurement noise on training reward'
t5='Effect of action noise on training reward (std in %)'
t6='Effect of initialisation on training reward'
from custom_callbacks import plot_results
sns.set_context("paper")
sns.set_style("whitegrid")
# sns.set(style='ticks',rc={"font.size": 10, 'font.family': ['sans-serif'], 'axes.grid': True, 'font.sans-serif': 'Times New Roman'})

'''
PLOT THE TRAINING reward from csv log, namely monitor files
'''
if PLOT_TRAINING_REWARD:
    xArr,yArr,legs=plot_results('./EJPH/tension-perf',title=t1,only_return_data=True) #'Effect of varying tension on the learning'
    legs=[leg+'V' for leg in legs[:,-3]]
    save_show_fig(xArr,yArr,legs,title=t1,savename='./EJPH/plots/tension.pdf')

    xArr,yArr,legs=plot_results('./EJPH/static-friction',title=t2,only_return_data=True)
    legs=[round(float(leg[1:]),4) for leg in legs[:,-2]]
    save_show_fig(xArr,yArr,legs,title=t2,savename='./EJPH/plots/static.pdf')

    xArr,yArr,legs= plot_results('./EJPH/dynamic-friction',title=t3,only_return_data=True)
    legs=np.array([0,0.1,1,10])*0.1196
    save_show_fig(xArr,yArr,legs,title=t3,savename='./EJPH/plots/dynamic.pdf')

    xArr,yArr,legs= plot_results('./EJPH/encoder-noise',title=t4,only_return_data=True)
    legs=[round(float(leg),4) for leg in legs[:,-3]]
    save_show_fig(xArr,yArr,legs,title=t4,savename='./EJPH/plots/noise.pdf')

    xArr,yArr,legs= plot_results('./EJPH/action-noise',title=t5,only_return_data=True)
    legs=[0,0.1,1,10]
    save_show_fig(xArr,yArr,legs,title=t5,savename='./EJPH/plots/action_noise.pdf')

    xArr,yArr,legs= plot_results('./EJPH/experimental-vs-random',title=t6, only_return_data=True)
    legs=['experimental','random']
    save_show_fig(xArr,yArr,legs,title=t6,savename='./EJPH/plots/exp-vs-rand.pdf')
'''
PLOT THE inference reward from .npz, namely EvalCallback logs
'''
if PLOT_EVAL_REWARD:
    dirTension='./EJPH/tension-perfomance'
    dirStatic='./EJPH/static-friction'
    dirDynamic='./EJPH/dynamic-friction'
    dirNoise='./EJPH/noise-eval'
    dirAction='./EJPH/plots/action-eval'
    dirReset='./EJPH/plots/exp-vs-rand-eval'

    # dirTension='./EJPH/plots/tension-eval'
    # dirStatic='./EJPH/plots/static-eval'
    # dirNoise='./EJPH/plots/noise-eval'
    # dirAction='./EJPH/plots/action-eval'
    # dirReset='./EJPH/plots/exp-vs-rand-eval'
    NUM_TIMESTEPS = 100000
    EVAL_NUM_STEPS = 5000
    timesteps = np.linspace(EVAL_NUM_STEPS, NUM_TIMESTEPS, int(NUM_TIMESTEPS / EVAL_NUM_STEPS))
    def plot_from_npz(path,xlabel,ylabel,title):
        filenames = sorted(glob.glob(path + '/*.npz'))
        for filename in filenames:
            data = np.load(filename)
            meanRew, stdRew = np.mean(data["results"], axis=1), np.std(data["results"], axis=1, keepdims=False)
            plt.plot(timesteps, meanRew)
            plt.fill_between(timesteps, meanRew + stdRew, meanRew - stdRew, alpha=0.2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
        return filenames
    xl='Timesteps'
    yl='Rewards'
    title='Effect of applied tension on the "greedy policy" reward'
    filenames=plot_from_npz(dirTension,xl,yl,title)
    legends=generate_legends(filenames)
    legs = []
    for i, counter in enumerate(legends[:, -2]):
        legs.append(legends[i, -3] + legends[i, -2])
    plt.legend(legs)
    plt.savefig('./EJPH/plots/greedy_tension.pdf')
    plt.show()


    title='Effect of static friction on the "greedy policy" reward'
    filenames=plot_from_npz(dirStatic,xl,yl,title)
    legends = generate_legends(filenames)
    legs=[round(float(leg[1:]),4) for leg in legends[:,-2]]
    plt.legend(legs)
    plt.savefig('./EJPH/plots/greedy_static.pdf')
    plt.show()

    title='Effect of viscous friction on the "greedy policy" reward'
    filenames=plot_from_npz(dirDynamic,xl,yl,title)
    legs=np.array([0,0.1,1,10])*0.1196
    plt.legend(legs)
    plt.savefig('./EJPH/plots/greedy_viscous.pdf')
    plt.show()

    data = np.load('./EJPH/experimental-vs-random/random.npz')
    data2 = np.load('./EJPH/experimental-vs-random/experimental.npz')
    meanRew=np.mean(data["results"],axis=1)
    meanRew2=np.mean(data2["results"],axis=1,keepdims=False)
    stdRew=np.std(data["results"],axis=1)
    stdRew2=np.std(data2["results"],axis=1)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    plt.plot(timesteps,meanRew, 'ro-')
    plt.fill_between(timesteps,meanRew + stdRew, meanRew - stdRew, facecolor='red', alpha=0.2)
    plt.plot(timesteps,meanRew2, 'bo--')
    plt.fill_between(timesteps,meanRew2 + stdRew2, meanRew2 - stdRew2, facecolor='blue', alpha=0.2)
    plt.xlabel('timesteps')
    plt.ylabel('Rewards')
    plt.title('Effect of initialisation on the "greedy policy" reward from experimental state')#random
    plt.legend(['random','experimental'])
    plt.savefig('./EJPH/exp-vs-rand.pdf')
    plt.show()


