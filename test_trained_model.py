#tensorboard --logdir ./sac_cartpole_tensorboard/
import gym
import numpy as np
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC,DDPG,TD3, DQN
from env_custom import CartPoleCusBottom,CartPoleCusBottomNoisy, CartPoleEnv,CartPoleFriction, CartPoleCosSin
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import time
Te=2e-2
# env = CartPoleCusBottom()
env= CartPoleCosSin() #CartPoleFriction()#CartPoleCusBottom()
env.MAX_STEPS_PER_EPISODE = 10000
# Load the saved statistics
env = DummyVecEnv([lambda: env])
env = VecNormalize.load('envNorm.pkl', env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

model = SAC.load("./logs/best_model", env=env)
obs = env.reset()

while True:
    action,_states=model.predict(obs,deterministic=True)
    obs, rewards, dones, _ = env.step(action)
    env.render()
    time.sleep(Te)
    if dones:
        env.reset()
