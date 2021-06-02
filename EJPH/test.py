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

if PLOT_EVAL_REWARD:
    dirTension='./EJPH/tension'
    dirStatic='./EJPH/static-friction'
    dirNoise='./EJPH/noise-eval'
    dirAction='./EJPH/plots/action-eval'
    dirReset='./EJPH/plots/exp-vs-rand-eval'

    dirTension='./EJPH/plots/tension-eval'
    dirStatic='./EJPH/plots/static-eval'
    dirNoise='./EJPH/plots/noise-eval'
    dirAction='./EJPH/plots/action-eval'
    dirReset='./EJPH/plots/exp-vs-rand-eval'

    data = np.load('./EJPH/experimental-vs-random/random.npz')
    data2 = np.load('./EJPH/experimental-vs-random/experimental.npz')

    meanRew=np.mean(data["results"],axis=1)
    meanRew2=np.mean(data2["results"],axis=1,keepdims=False)
    stdRew=np.std(data["results"],axis=1)
    stdRew2=np.std(data2["results"],axis=1)
    sns.set_context("paper")
    sns.set_style("whitegrid")


    NUM_TIMESTEPS=90000
    EVAL_NUM_STEPS=15000
    timesteps=np.linspace(EVAL_NUM_STEPS,NUM_TIMESTEPS,int(NUM_TIMESTEPS/EVAL_NUM_STEPS))
    plt.plot(timesteps,meanRew, 'ro-')
    plt.fill_between(timesteps,meanRew + stdRew, meanRew - stdRew, facecolor='red', alpha=0.2)
    plt.plot(timesteps,meanRew2, 'bo--')
    plt.fill_between(timesteps,meanRew2 + stdRew2, meanRew2 - stdRew2, facecolor='blue', alpha=0.2)
    plt.xlabel('Tension (V)')
    plt.ylabel('Rewards')
    plt.title('Effect of initialisation (random vs experimental) on the "greedy policy" reward from experimental state')#random
    plt.savefig('./EJPH/exp-vs-rand.pdf')
    plt.show()



