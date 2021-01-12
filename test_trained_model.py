#tensorboard --logdir ./sac_cartpole_tensorboard/
import gym
import numpy as np
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC,DDPG,TD3
from env_custom import CartPoleCus, CartPoleCusBottom,CartPoleCusBottomNoisy
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
import argparse
# env=CartPoleCusBottomNoisy()
env=CartPoleCusBottom()
env=env.unwrapped
env.MAX_STEPS_PER_EPISODE=5000

model = SAC.load("./logs/best_model", env=env)
# model = DDPG.load("./logs/best_model", env=env)
# model = TD3.load("./logs/best_model", env=env)
parser = argparse.ArgumentParser(description='deepRL for inverted pendulum')
parser.add_argument('--env',default='CartPoleCusBottomNoisy',type=str)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()