'''
I3S lab:
By defaut the agent in trained and inference test is recorded at the end, results of an inference are recorded to .npz
If the WEIGHTS variable is not None, we try to load the selected weights to the model.

'''
import sys
import os
import torch
import re
from distutils.dir_util import copy_tree
sys.path.append(os.path.abspath('./..'))
sys.path.append(os.path.abspath('./'))
from custom_callbacks import plot_results
from env_wrappers import Monitor
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import EvalCustomCallback, CheckPointEpisode
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from tcp_envV2 import CartPoleZmq
from utils import read_hyperparameters
from pathlib import Path
from pendule_pi import PendulePy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from glob import glob
import numpy as np
import time
#Simulation parameters
Te = 0.05 #sampling time
EP_STEPS = 800 #num steps in an episode
STEPS_TO_TRAIN = 150000
PWM = 151 #PWM command to apply 0-255
INFERENCE_STEPS = 800 #steps to test the model

TRAIN = False#True #if true train, else only inf
x_threshold = 0.33 #limit on cart total: 33*2+5*2(hard)+4*2(soft) = 84 <84.5(rail)
MANUAL_SEED = 5


#paths to save monitor, models...
log_save = f'./weights/dqn50-real/pwm{PWM}'
Path(log_save).mkdir(parents=True, exist_ok=True)
WEIGHTS = None#f'./weights/dqn50-real/pwm{PWM}/dqn_rpi.zip'#None#f'./weights/dqn50-real/pwm{PWM}/dqn_rpi.zip'
REPLAY_BUFFER_WEIGHTS = None#f'./weights/dqn50-real/pwm{PWM}/dqn_rpi_buffer.pkl'  #None
logdir = f'./weights/dqn50-real/pwm{PWM}'
#initialisaiton of a socket and a gym env
pendulePy = PendulePy(wait=5, host='rpi5') #host:IP_ADRESS
env = CartPoleZmq(pendulePy=pendulePy, x_threshold=x_threshold, max_pwm = PWM)
torch.manual_seed(MANUAL_SEED)



