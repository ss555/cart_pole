import gym
import numpy as np
import yaml
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from env_custom import CartPoleCusBottomDiscrete, CartPoleDiscrete2actions
from gym.wrappers import TimeLimit
from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


env = CartPoleDiscrete2actions()
## Automatically normalize the input features and reward
env1=DummyVecEnv([lambda:env])
#env=env.unwrapped
env=VecNormalize(env1,norm_obs=True,norm_reward=True,clip_obs=10000,clip_reward=10000)
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env, best_model_save_path='stable_baselines_dqn_best/',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

model = DQN(MlpPolicy, env, gamma=1, batch_size=4096, learning_rate=linear_schedule(0.0003), policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log='./dqn_tensorboard/')
model.learn(total_timesteps=200000, callback=eval_callback, tb_log_name="normal")

model.save("dqn_cartpole")
model.load('./stable_baselines_dqn_best/best_model')
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()