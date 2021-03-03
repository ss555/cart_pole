import gym
import numpy as np
from sb3_contrib import TQC
from env_custom import CartPoleCusBottom
from typing import Callable
from stable_baselines3.common.callbacks import EvalCallback

env = CartPoleCusBottom()

# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env, best_model_save_path='../stable_baselines_dqn_best/',
                             log_path='../logs/', eval_freq=5000,
                             deterministic=True, render=False)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


policy_kwargs = dict(n_critics=2, n_quantiles=25)
model = TQC("MlpPolicy", env, learning_rate=linear_schedule(0.0003), batch_size=512, top_quantiles_to_drop_per_net=2, verbose=0, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=100000, log_interval=4, callback=[eval_callback])

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()