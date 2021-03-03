import gym
import pybullet_envs
import torch
import gym
import numpy as np
import os
from stable_baselines3 import SAC, DDPG, TD3, A2C
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from typing import Callable
from custom_callbacks import plot_results
from stable_baselines3.sac.policies import MlpPolicy

logdir='./logs/training'
env0 = gym.make('InvertedPendulumSwingupBulletEnv-v0')
env0 = Monitor(env0, logdir)
## Automatically normalize the input features and reward
env1=DummyVecEnv([lambda:env0])
#env=env.unwrapped
env=VecNormalize(env1,norm_obs=True,norm_reward=True,clip_obs=10000,clip_reward=10000)
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)

STEPS_TO_TRAIN = 1000

# Stop training when the model reaches the reward threshold
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
logdir='./logs/training'
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env, best_model_save_path=logdir,
                             log_path=logdir, eval_freq=5000,# callback_on_new_best=callback_on_best,
                             deterministic=True, render=False)
manual_seed=5
env.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
model = SAC.load(logdir+'/best_model')
model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100,  # tb_log_name="normal",
            callback=[eval_callback, callbackSave])
plot_results(logdir, title='Learning Curve'+manual_seed)

#WHEN NORMALISING
env.save(logdir+'/envNorm.pkl')
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False
plot_results(logdir)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()