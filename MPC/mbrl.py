import sys
import os
sys.path.append(os.path.abspath('./'))
from utils import linear_schedule
from custom_callbacks import plot_results,CheckPointEpisode
# from env_wrappers import Monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback, EvalCustomCallback, EvalThetaDotMetric
from env_custom import CartPoleButter, CartPoleDebug, CartPoleDiscreteHistory#,CartPoleContinous,CartPoleDiscreteHistory#,CartPoleDiscreteButter2
import argparse
import numpy as np
from utils import read_hyperparameters
from pathlib import Path

#learn model train through imagination