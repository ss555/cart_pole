#tensorboard --logdir ./sac_cartpole_tensorboard/
import torch
import gym
import numpy as np
import os
from stable_baselines3 import SAC, DDPG, TD3, A2C
from env_custom import CartPoleCusBottom, CartPoleCusBottomNoisy, CartPoleFriction, CartPoleCosSin
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#from stable_baselines3.common import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from typing import Callable

env = CartPoleCosSin(Te=0.05)#CartPoleCusBottom(Te=0.05) #CartPoleFriction()#
env = Monitor(env, log_dir)
## Automatically normalize the input features and reward
env1=DummyVecEnv([lambda:env])
#env=env.unwrapped
env=VecNormalize(env1,norm_obs=True,norm_reward=False,clip_obs=10000,clip_reward=10000)

# Stop training when the model reaches the reward threshold
#callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
							 log_path='./logs/', eval_freq=5000, #callback_on_new_best=callback_on_best,
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


	RL_ALGO=args.algo

	STEPS_TO_TRAIN = args.steps
	update_freq = 1  # for the q-target network
	if (RL_ALGO == 'SAC'):
		from stable_baselines3.sac.policies import MlpPolicy#learning_rate=0.0001,
		print('SAC chosen')
		manual_seed=4
		env.seed(manual_seed)
		torch.manual_seed(manual_seed)
		np.random.seed(manual_seed)
		#model=SAC(MlpPolicy,env=env,**sac_kwargs)
		# model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(0.0003), ent_coef='auto', #action_noise=action_noise,
		# 			batch_size=2048, use_sde=True, buffer_size=300000,
		# 			learning_starts=30000, tensorboard_log="./sac_cartpole_tensorboard",
		# 			policy_kwargs=dict(net_arch=[256, 256]), train_freq=-1, n_episodes_rollout=1, gradient_steps=-1)
		#pybullet conf n_timesteps: !!float 3e5
		# model = SAC(MlpPolicy, env=env, learning_rate=float(7.3e-4),buffer_size=300000,
  		# 		batch_size= 256, ent_coef= 'auto', gamma= 0.98, tau=0.02, train_freq= 64,  gradient_steps= 64,learning_starts= 10000,
  		# 		use_sde= True, policy_kwargs= dict(log_std_init=-3, net_arch=[400, 300]))
		model = SAC(MlpPolicy, env=env, learning_rate=float(7.3e-4), buffer_size=300000,
					batch_size=256, ent_coef='auto', gamma=0.98, tau=0.02, n_episodes_rollout=-1, gradient_steps=-1,
					learning_starts=10000,
					use_sde=True, policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]))


	# model for pendulum starting from bottom
	with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
		model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100, tb_log_name="normal",
					callback=[cus_callback, eval_callback])

		#WHEN NORMALISING
		env.save('envNorm.pkl')
		env.training = False
		# reward normalization is not needed at test time
		env.norm_reward = False
	# model.save(RL_ALGO+args.env)

	# Don't forget to save the VecNormalize statistics when saving the agent
	log_dir = "/tmp/"
	model.save(log_dir + "cartpole.pkl")
	model.save_replay_buffer("sac_swingup.pkl")
	obs = env.reset()
	while True:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		env.render()
		if dones:
			env.reset()