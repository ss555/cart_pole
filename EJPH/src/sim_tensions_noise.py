import sys
import os
import numpy as np
import time
import subprocess
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
sys.path.append(os.path.abspath('./../..'))
from utils import linear_schedule, plot
from custom_callbacks import plot_results
from env_wrappers import Monitor
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, SAC
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

#TODO use subprocess to parallelise sim
STEPS_TO_TRAIN = 150000
EP_STEPS = 800
Te = 0.05
MANUAL_SEED = 1
video_folder = None
# simulation results
#Done episode reward for seed 0,5 + inference
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
hyperparams = read_hyperparameters('dqn_cartpole_50')

# DONE temps d’apprentissage et note en fonction du coefficient de friction statique 4 valeurs du coefficient:Ksc,virt= 0,0.1∗Ksc,manip,Ksc,manip,10∗Ksc,manipDiscussion
STATIC_FRICTION_CART = 1.166390864012042
# STATIC_FRICTION_ARR = np.array([200]) * STATIC_FRICTION_CART #150 not working
STATIC_FRICTION_ARR = np.array([0, 0.1, 1, 10]) * STATIC_FRICTION_CART

# DONE temps d’apprentissage et note en fonction de l’amplitude du controle
TENSION_RANGE = np.arange(6.5,7.1,0.1)#
# DONE temps  d’apprentissage  et  note  en  fonction  du  coefficient  de frottement dynamique
DYNAMIC_FRICTION_PENDULUM = 0.07035332644615992
DYNAMIC_FRICTION_ARR = np.array([0, 0.1, 1, 10]) * DYNAMIC_FRICTION_PENDULUM

# NOISE_TABLE = np.array([5, 10]) * np.pi / 180
NOISE_TABLE = np.array([0, 0.01, 0.05, 0.1, 0.15, 0.5, 1, 5, 10]) * np.pi / 180


# DONE graphique la fonction de recompense qui depends de la tension a 40000 pas
# DONE valeur de MAX recompense en fonction de tension
logdir = './EJPH/tensions-noise'
Path(logdir).mkdir(parents=True, exist_ok=True)
filenames = []
# train to generate data
# inference to test the models
# rainbow to plot in inference at different timesteps
for i, tension in enumerate(TENSION_RANGE):
    env = CartPoleRK4(Te=Te, integrator='rk4', forceStd=1, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
    envEval = CartPoleRK4(Te=Te, integrator='rk4', forceStd=1, N_STEPS=EP_STEPS, tensionMax=tension, resetMode='experimental')
    filename = logdir + f'/_tension_sim_{tension}_V_'
    env = Monitor(env, filename=filename)
    model = DQN(env=env, **hyperparams, seed=MANUAL_SEED)
    eval_callback = EvalThetaDotMetric(envEval, save_model=False, log_path=filename, eval_freq=5000, deterministic=True)
    print(f'simulation for {tension} V')
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])

    #