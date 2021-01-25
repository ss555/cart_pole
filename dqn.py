import gym
import numpy as np
import yaml
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from env_custom import CartPoleCusBottomDiscrete
env = CartPoleCusBottomDiscrete()
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env, best_model_save_path='stable_baselines_dqn_best/',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False)
# Custom MLP policy of two layers of size 32 each with Relu activation function
model = DQN(MlpPolicy, env, batch_size=4096, policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log='./dqn_tensorboard/')
model.learn(total_timesteps=1000000, callback=eval_callback)

model.save("dqn_cartpole")
model.load('./stable_baselines_dqn_best/best_model')
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()