#tensorboard --logdir ./sac_cartpole_tensorboard/
import torch
import gym
import numpy as np
import os
from stable_baselines3 import SAC, DDPG, TD3, A2C
from env_custom import CartPoleCusBottom,CartPoleCusBottomNoisy
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#from stable_baselines3.common import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
from typing import Callable
env=CartPoleCusBottom()
env=env.unwrapped
# Automatically normalize the input features and reward
envTest=
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
							 log_path='./logs/', eval_freq=5000,
							 deterministic=True, render=False)
#arguments
parser = argparse.ArgumentParser(description='deepRL for inverted pendulum')
parser.add_argument('--algo', metavar="SAC",type=str,
					default='SAC',
					help='rl algorithm: SAC, DDPG, TD3, A2C')
parser.add_argument('--env', default="cartpole_bottom",type=str,
					metavar="cartpole",
					help='rl env: cartpole_bottom,cartpole_balance, ')
parser.add_argument('--steps', default=100000, type=int, help='timesteps to train')
args = parser.parse_args()
if __name__ == '__main__':
	#lr schedule
	def linear_schedule(initial_value: float) -> Callable[[float], float]:
		def func(progress_remaining: float) -> float:
			return progress_remaining * initial_value

		return func


	from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
	# The noise objects for DDPG
	n_actions = env.action_space.shape[-1]
	action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

	RL_ALGO=args.algo

	STEPS_TO_TRAIN = args.steps
	update_freq = 1  # for the q-target network
	sac_kwargs = {'gamma': 0.98, 'learning_rate': 0.00275211369549068, 'batch_size': 1024, 'buffer_size': 1000000, 'learning_starts': 10000, 'train_freq': 256, 'tau': 0.01, 'policy_kwargs': dict(log_std_init=0.5745995200274363, net_arch=[256, 256])}
	if (RL_ALGO == 'SAC'):
		from stable_baselines3.sac.policies import MlpPolicy#learning_rate=0.0001,
		#model = SAC(MlpPolicy, env=env, **sac_kwargs)
		# model=SAC(MlpPolicy, env=env, gamma= 0.98, learning_rate= 0.00275211369549068,
		# batch_size: 1024, buffer_size: 1000000, learning_starts: 10000, 'train_freq': 256, 'tau': 0.01, 'log_std_init': 0.5745995200274363, 'net_arch': 'medium')
		print('SAC chosen')
		env.seed(2)
		torch.manual_seed(2)
		np.random.seed(2)
		#model=SAC(MlpPolicy,env=env,**sac_kwargs)
		model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(0.0003), ent_coef='auto',
					verbose=0, action_noise=action_noise,
					batch_size=2048, buffer_size=50000,
					learning_starts=30000, policy_kwargs=dict(net_arch=[256, 256]),
					tensorboard_log="./sac_cartpole_tensorboard", train_freq=-1, n_episodes_rollout=1, gradient_steps=2048)

	elif (RL_ALGO == 'DDPG'):
		from stable_baselines.ddpg.policies import MlpPolicy
		# action noise
		n_actions = env.action_space.shape[-1]
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

		# using parameter noise
		DDPG(policy=MlpPolicy, env=env, param_noise=param_noise, verbose=2, batch_size=512,
			 policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log="./ddpg_cartpole_tensorboard/")

	elif (RL_ALGO == 'TD3'):
		from stable_baselines.td3.policies import MlpPolicy
		TD3(MlpPolicy, env=env, action_noise=action_noise, batch_size=512, learning_starts=100000,
			tensorboard_log="./td3_cartpole_tensorboard/")

	elif (RL_ALGO == 'A2C'):
		env = make_vec_env(env, n_envs=8)
		model = A2C(policy=CustomDDPGPolicy, gamma=0.95, env=env, ent_coef=0.0, verbose=2,
					tensorboard_log="./ddpg_cartpole_tensorboard/")
	else:
		print('this algo is not included, please choose from SAC, DDPG, TD3, A2C')

	# model for pendulum starting from bottom
	with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
		model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100, tb_log_name="normal",
					callback=[cus_callback, eval_callback])

	# model.save(RL_ALGO+args.env)

	# Don't forget to save the VecNormalize statistics when saving the agent
	log_dir = "/tmp/"
	model.save(log_dir + "cartpole")
	model.save_replay_buffer("sac_swingup")
	obs = env.reset()
	while True:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		env.render()
		if dones:
			env.reset()