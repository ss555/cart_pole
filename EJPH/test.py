import sys
import os
import numpy as np
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule, plot_results
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
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
STEPS_TO_TRAIN=100000
EP_STEPS=800
Te=0.05
from custom_callbacks import plot_results
plot_results('./EJPH')
