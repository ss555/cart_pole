#tensorboard --logdir ./sac_cartpole_tensorboard/
import torch
import gym
import numpy as np
import os
from typing import Callable
from stable_baselines3 import SAC, DDPG, TD3, A2C
from env_custom import CartPoleCusBottom,CartPoleCusBottomNoisy
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.sac.policies import MlpPolicy
STEPS_TO_TRAIN=100000

env=CartPoleCusBottomNoisy(std_masspole=1e-3,std_masscart=1e-2)
env= env.unwrapped

#lr schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
	def func(progress_remaining: float) -> float:
		return progress_remaining * initial_value

	return func
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env, best_model_save_path='./logs/noisy',
							 log_path='./logs/noisy', eval_freq=5000,
							 deterministic=True, render=False)
print('SAC chosen')
env.seed(2)
torch.manual_seed(2)
np.random.seed(2)
# model=SAC(MlpPolicy,env=env,**sac_kwargs)
model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(0.0003), ent_coef='auto',
			verbose=0,  # action_noise=action_noise,
			batch_size=2048,
			learning_starts=30000, policy_kwargs=dict(net_arch=[256, 256]),
			tensorboard_log="./sac_cartpole_tensorboard")  # , train_freq=-1, n_episodes_rollout=1, gradient_steps=-1)

# model for pendulum starting from bottom
with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
	model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100, tb_log_name="noisy",
				callback=[cus_callback, eval_callback])