'''
records the video of the training on openai gym environement
DOESN'T work with the non-registered environements.
'''

import sys
import os
# sys.path.append(os.path.abspath('./..'))
import gym
import cartpole
sys.path.append(os.path.abspath('./'))
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv,SubprocVecEnv
from src.env_custom import CartPoleRK4
from src.utils import linear_schedule
from src.custom_callbacks import plot_results
# from src.env_wrappers import Monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from src.custom_callbacks import EvalCustomCallback
from src.custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from src.env_custom import CartPoleRK4 #,CartPoleContinous,CartPoleDiscreteHistory#,CartPoleDiscreteButter2
import argparse
from src.utils import read_hyperparameters
from pathlib import Path
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback






if __name__=='__main__':
    EP_STEPS = 10
    TRAIN = True

    Te = 0.05
    # eval_env = CartPoleRK4()
    eval_env = gym.make('cartpoleSwingD-v0')
    eval_env = SubprocVecEnv([lambda: eval_env])
    num_steps = 1e3
    prefix = 'dafsdfasdf'
    video_folder = './logs/video'
    # record_video_learning(eval_env, model, video_length=EP_STEPS, video_folder='./logs/video')
    os.makedirs(video_folder,exist_ok=True)
    # # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder, record_video_trigger=lambda step: step == 0, video_length=num_steps, name_prefix=prefix)
    if TRAIN:
        hyperparams = read_hyperparameters('dqn_50')
        model = DQN(env=eval_env, **hyperparams)
    else:
        path_weights = './weights/dqn50-sim/best_model'
        model = DQN.load(path_weights)
        model.env = eval_env
    checkpoint = CheckpointCallback(save_freq=10000, save_path=video_folder)
    with ProgressBarManager(num_steps) as cus_callback:
        model.learn(total_timesteps=num_steps, callback=[cus_callback, checkpoint])
    # Close the video recorder
    eval_env.close()


    # play(eval_env_id='cartpoleSwingD-v0', model=model, video_path='./logs/video/dqn_model.mp4')