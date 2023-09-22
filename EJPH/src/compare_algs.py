'''
simulates different algorithms on the same environment
'''
import sys
import os
import numpy as np
import time
import subprocess
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from src.utils import linear_schedule, plot
from src.custom_callbacks import plot_results
from rlutils.env_wrappers import Monitor
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

#TODO use subprocess to parallelise sim

STEPS_TO_TRAIN = 150000
EP_STEPS = 800
Te = 0.05
MANUAL_SEED = 1
video_folder = None
# simulation results
DYNAMIC_FRICTION_SIM = False  # True
STATIC_FRICTION_SIM = False
encNoiseVarSim = False
ACTION_NOISE_SIM = False #False
RESET_EFFECT = False  # True#False
EVAL_TENSION_FINAL_PERF = True  # evaluate final PERFORMANCE of a cartpole for different voltages
SEED_TRAIN = False
# other
PLOT_FINAL_PERFORMANCE_STD = False  # False#
qLearningVsDQN = False  # compare q-learn and dqn
EVAL_TENSION_FINAL_PERF_seed = False  # evaluate final PERFORMANCE of a cartpole for different voltages
logdir = './EJPH/'
DISCRETE_SIM = False#True
# DISCRETE_SIM = True

if DISCRETE_SIM:
    #discr
    hyperparams = read_hyperparameters('dqn_cartpole_50')
    hyperparamsPpo = read_hyperparameters('ppo_disc_cartpole_50')

    tension=12
    cart_continuous = CartPoleRK4(Te=Te, discreteActions=False, sparseReward=False, tensionMax=12)
    cart_discr =      CartPoleRK4(Te=Te, discreteActions=True, sparseReward=False, tensionMax=12)
    for ALGO, hyperparamsl in zip([PPO, DQN, DDQN],[hyperparamsPpo,hyperparams,hyperparams]):
        env = CartPoleRK4(Te=Te, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
        envEval = CartPoleRK4(Te=Te, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
        filename = os.path.join(logdir , f'comapre/discr_sim_{ALGO.__name__}/')
        os.makedirs(filename, exist_ok=True)
        if video_folder is not None:
            env = VideoRecorderWrapper(env, video_folder=video_folder, record_video_trigger=lambda step: step == 0, video_length=STEPS_TO_TRAIN, name_prefix=filename)
            envEval = VideoRecorderWrapper(envEval, video_folder=video_folder, record_video_trigger=lambda step: step == 0, video_length=STEPS_TO_TRAIN, name_prefix=filename)
        env = Monitor(env, filename=filename)
        model = ALGO('MlpPolicy', env=env, seed=MANUAL_SEED,**hyperparamsl)
        eval_callback = EvalThetaDotMetric(envEval, log_path=os.path.join(filename,'inference.zip'), eval_freq=5000, deterministic=True)
        print(f'simulation for {ALGO} V')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])

#continous
hyperparams = read_hyperparameters('sac_cartpole_50')
# hyperparams = read_hyperparameters('sac_cartpole_50_old')
hyperparamsPpo = read_hyperparameters('ppo_cont_cartpole_50')
tension=12

for ALGO, hyperparamsl in zip([PPO],[hyperparamsPpo]):
# for ALGO, hyperparamsl in zip([SAC],[hyperparamsPpo,hyperparams]):
    env = CartPoleRK4(Te=Te, discreteActions=False, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
    envEval = CartPoleRK4(Te=Te, discreteActions=False, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
    filename = os.path.join(logdir , f'comapre/cont_sim_{ALGO.__name__}/')
    os.makedirs(filename, exist_ok=True)
    if video_folder is not None:
        env = VideoRecorderWrapper(env, video_folder=video_folder, record_video_trigger=lambda step: step == 0, video_length=STEPS_TO_TRAIN, name_prefix=filename)
        envEval = VideoRecorderWrapper(envEval, video_folder=video_folder, record_video_trigger=lambda step: step == 0, video_length=STEPS_TO_TRAIN, name_prefix=filename)
    env = Monitor(env, filename=filename)
    model = ALGO('MlpPolicy', tensorboard_log=logdir, env=env, seed=MANUAL_SEED,**hyperparamsl)
    eval_callback = EvalThetaDotMetric(envEval, log_path=os.path.join(filename,'inference.zip'), eval_freq=5000, deterministic=True)
    print(f'simulation for {ALGO} V')
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])

#todo plot inference/ inf path i.npz