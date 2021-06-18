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
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'

def play(env, model, steps: int = 50, deterministic: bool =True, video_path:str='./logs/video/dqn.mp4'):
    num_episodes = 0
    video_recorder = None
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

def record_video(eval_env, model, video_length=500, prefix='dqn', video_folder='./logs/video'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
    # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder, record_video_trigger=lambda step: step == 0, video_length=video_length, name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action = [1]
    # action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()

if __name__=='__main__':
    EP_STEPS = 100
    path_weights = './weights/dqn50-sim/best_model.zip'
    model = DQN.load(path_weights)
    Te = 0.05
    eval_env = CartPoleButter(Te=Te, x_threshold=0.35, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706,
                         resetMode='experimental', sparseReward=False, Km=0.0,
                         n=1)  # ,integrator='ode')#,integrator='rk4')

    # eval_env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
    env0 = gym.make('cartpoleSwingD-v0')
    eval_env = DummyVecEnv([lambda: env0])
    # record_video(eval_env, model, video_length=EP_STEPS, video_folder='./logs/video')
    play(eval_env, model, video_path='./logs/video/dqn_model.mp4')