#tensorboard --logdir ./sac_cartpole_tensorboard/
import torch
import numpy as np
from stable_baselines3 import SAC
from env_custom import CartPoleButter#, CartPoleCosSinTension #CartPoleCosSinT_10
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
from typing import Callable
from custom_callbacks import plot_results
from utils import linear_schedule, plot
from stable_baselines3.common.noise import NormalActionNoise
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
NORMALISE=False
EP_STEPS = 800#*2
MANUAL_SEED = 5
logdir='./logs/sac/'
# env = CartPoleCosSinTension(Te=0.05)#
env = CartPoleButter(Te=0.02,discreteActions=False,N_STEPS=EP_STEPS,f_a=-7.794018686563599, f_b=0.37538450501353504, f_c=-0.4891760779740128, f_d=-0.002568958116514183,sparseReward=False,tensionMax=12)#integrator='rk4')#
env0 = Monitor(env, logdir)
## Automatically normalize the input features and reward
env=DummyVecEnv([lambda:env0])

if NORMALISE:
	env = VecNormalize.load('envNorm.pkl', env)
	# env=VecNormalize(env1, norm_obs=True, norm_reward=True, clip_obs=10000, clip_reward=10000)
	env.training = True
	envEval=env
	envEval.training=False
	envEval.norm_reward=True
	# envEval=VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=10000, clip_reward=10000)
else:
	envEval=env

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=3800, verbose=1)

# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env,
							 best_model_save_path=logdir,
							 log_path=logdir, eval_freq=5000, callback_on_new_best=callback_on_best,
							 deterministic=True, render=False)
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir, monitor_filename=logdir+'monitor.csv')
#arguments
parser = argparse.ArgumentParser(description='deepRL for inverted pendulum')
parser.add_argument('--algo', metavar="SAC",type=str,
					default='SAC',
					help='rl algorithm: SAC, DDPG, TD3, A2C')
parser.add_argument('--env', default="cartpole_bottom",type=str,
					metavar="cartpole",
					help='rl env: cartpole_bottom,cartpole_balance, ')
parser.add_argument('--steps', default=240000, type=int, help='timesteps to train')
args = parser.parse_args()

if __name__ == '__main__':
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
		# ##sde
		# from parameters import sac_cartpole_20
		from utils import read_hyperparameters
		hyperparams = read_hyperparameters('sac_cartpole_20')
		# from parameters import dqn_sim50
		model = SAC(env=env, **hyperparams, seed=MANUAL_SEED)
	try:
		# model for pendulum starting from bottom
		with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
			model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100, #tb_log_name="normal",
						callback=[cus_callback, eval_callback, callbackSave])
			if NORMALISE:
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
		model.save(logdir + "cartpole.pkl")
		model.save_replay_buffer("sac_swingup_simulation.pkl")

