#tensorboard --logdir ./sac_cartpole_tensorboard/
import torch
import gym
import numpy as np
import os
from stable_baselines3 import SAC
from env_custom import CartPoleCosSinDev, CartPoleCosSinTension #CartPoleCosSinT_10
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from typing import Callable
from custom_callbacks import plot_results

logdir='./logs/'
# plot_results(logdir)
# env00 = CartPoleCosSinDev()#
env00 = CartPoleCosSinTension(Te=0.05)#
env0 = Monitor(env00, logdir)
## Automatically normalize the input features and reward
env1=DummyVecEnv([lambda:env0])
#env=env.unwrapped
env = VecNormalize.load('envNorm.pkl', env1)
env.training=True
# env=VecNormalize(env1, norm_obs=True, norm_reward=True, clip_obs=10000, clip_reward=10000, gamma=0.98)
# envEval=VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=10000, clip_reward=10000)
envEval = VecNormalize.load('envNorm.pkl', env1)
envEval.training=False
# Stop training when the model reaches the reward threshold
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=2400, verbose=1)

# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(envEval, best_model_save_path='./logs/',
							 log_path=logdir, eval_freq=5000, callback_on_new_best=callback_on_best,
							 deterministic=True, render=False)
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
#arguments
parser = argparse.ArgumentParser(description='deepRL for inverted pendulum')
parser.add_argument('--algo', metavar="SAC",type=str,
					default='SAC',
					help='rl algorithm: SAC, DDPG, TD3, A2C')
parser.add_argument('--env', default="cartpole_bottom",type=str,
					metavar="cartpole",
					help='rl env: cartpole_bottom,cartpole_balance, ')
parser.add_argument('--steps', default=300000, type=int, help='timesteps to train')
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
		manual_seed=5
		#env.seed(manual_seed) #make reset from the same point-deterministic
		torch.manual_seed(manual_seed)
		np.random.seed(manual_seed)

		#model=SAC(MlpPolicy,env=env,**sac_kwargs)
		# model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(0.0003), ent_coef='auto', #action_noise=action_noise,
		# 			batch_size=2048, use_sde=True, buffer_size=300000,
		# 			learning_starts=30000, tensorboard_log="./sac_cartpole_tensorboard",
		# 			policy_kwargs=dict(net_arch=[64, 64]), train_freq=-1, n_episodes_rollout=1, gradient_steps=-1)
		#pybullet conf n_timesteps: !!float 3e5
		##sde
		# model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(1e-3), buffer_size=300000,
  		# 		batch_size= 1024, ent_coef= 'auto', gamma= 0.98, tau=0.02, train_freq= 64,  gradient_steps= 64,learning_starts= 10000,
  		# 		use_sde= True, policy_kwargs= dict(log_std_init=-3, net_arch=[400, 300]))#dict(pi=[256, 256], vf=[256, 256])
		model = SAC(MlpPolicy, env=env, learning_rate=0.001, target_update_interval=4,
				batch_size= 256, train_freq= 1, learning_starts= 10000, gamma=0.999,
				use_sde= False, policy_kwargs=dict(net_arch=dict(pi=[128, 128], qf=[128, 128])))#dict(pi=[256, 256], vf=[256, 256])
		#resume training
		# model = SAC.load("./logs/best_model", env=env)
		# model.learning_starts = 0
		# model.batch_size = 256
		# model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(1e-3), buffer_size=300000,
		# 		batch_size=2048, ent_coef='auto', gamma=0.98, tau=0.02, train_freq=-1, gradient_steps=-1, n_episodes_rollout=1,
		# 		learning_starts=10000,
		# 		use_sde=True, policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]))
		##action_noise
		# n_actions = env.action_space.shape[-1]
		# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
		# model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(float(1e-3)), buffer_size=300000,
		# 			batch_size=256, ent_coef='auto1.0', gamma=0.98, tau=0.02, n_episodes_rollout=-1, gradient_steps=-1,
		# 			learning_starts=10000, seed=manual_seed, use_sde=True, policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]))

	try:
		# model for pendulum starting from bottom
		with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
			model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100, #tb_log_name="normal",
						callback=[cus_callback, eval_callback, callbackSave])

			#WHEN NORMALISING
			env.save('envNorm.pkl')
			env.training = False
			# reward normalization is not needed at test time
			env.norm_reward = False
			plot_results(logdir)
		# Don't forget to save the VecNormalize statistics when saving the agent
		log_dir = "./tmp/"
		model.save(log_dir + "cartpole.pkl")
		model.save_replay_buffer("sac_swingup_simulation.pkl")
		obs = env.reset()
		while True:
			action, _states = model.predict(obs)
			obs, rewards, dones, info = env.step(action)
			env.render()
			if dones:
				env.reset()
	finally:
		# WHEN NORMALISING
		env.save('envNorm.pkl')
		model.save(log_dir + "cartpole.pkl")
		model.save_replay_buffer("sac_swingup_simulation.pkl")