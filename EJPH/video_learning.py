'''
records the video of the training on openai gym
'''

import sys
import os
# sys.path.append(os.path.abspath('./..'))
import gym
import cartpole
sys.path.append(os.path.abspath('./'))
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from env_custom import CartPoleButter
from utils import linear_schedule
from custom_callbacks import plot_results
# from env_wrappers import Monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import EvalCustomCallback
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from env_custom import CartPoleButter, CartPoleDebug, CartPoleDiscreteHistory #,CartPoleContinous,CartPoleDiscreteHistory#,CartPoleDiscreteButter2
import argparse
from utils import read_hyperparameters
from pathlib import Path
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback

# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'

def play(eval_env_id, model, steps: int = 50, deterministic: bool =True, video_path:str='./logs/video/dqn.mp4'):
    num_episodes = 0
    video_recorder = None
    env0 = gym.make(eval_env_id)
    env = DummyVecEnv([lambda: env0])
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()
    for i in range(steps):
        env.unwrapped.render()
        video_recorder.capture_frame()
        action = model.predict(obs, deterministic=deterministic)
        obs, rew, done, info = env.step(action)
        if done:
            obs = env.reset()
    if video_recorder.enabled:
        # save video of first episode
        print("Saved video.")
        video_recorder.close()
        video_recorder.enabled = False




if __name__=='__main__':
    EP_STEPS = 10
    TRAIN=True

    Te = 0.05
    # env = CartPoleButter()
    env = gym.make('cartpoleSwingD-v0')
    eval_env = DummyVecEnv([lambda: env])
    num_steps = 6e4
    prefix = 'dqn-learn'
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