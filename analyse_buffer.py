import pickle as pkl
import sys
import os
sys.path.append(os.path.abspath('./'))
from utils import linear_schedule
from custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import EvalCustomCallback
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from env_custom import CartPoleButter
import argparse
from utils import read_hyperparameters
from pathlib import Path
#TODO limit on 2s motorIden
#TODO change legends on the graph 0.9 to 1.01 and Vk
#monitor
xArrEx, yArrEx, _ = plot_results('./EJPH/real-cartpole/dqn', only_return_data=True)
#buffer
LOAD_BUFFER_PATH = "./weights/dqn/dqn_pi_swingup_bufferN"
env0 = CartPoleButter()
hyperparams = read_hyperparameters('dqn_cartpole_50')
model = DQN(env=env0, **hyperparams)
model.load_replay_buffer(LOAD_BUFFER_PATH)
print('s')



