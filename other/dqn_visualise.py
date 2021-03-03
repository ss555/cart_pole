import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from torch import nn
from env_custom import CartPoleCusBottomDiscrete
env = CartPoleCusBottomDiscrete()
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)
model=DQN.load("stable_baselines_dqn_best/best_model")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
