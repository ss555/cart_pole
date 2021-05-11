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
NORMALISE=False
logdir='./logs/sac/'
# env = CartPoleCosSinTension(Te=0.05)#
env = CartPoleButter(Te=0.05,discreteActions=False,sparseReward=False,tensionMax=12)#integrator='rk4')#
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

# model=SAC.load(logdir + "cartpole.pkl",env=env)

# envEval = VecNormalize.load('envNorm.pkl', env1)
#envEval.training = False
# Stop training when the model reaches the reward threshold
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=2400, verbose=1)

# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env,
							 best_model_save_path=logdir,
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
		model = SAC(MlpPolicy, env=env, learning_rate=linear_schedule(1e-3), buffer_size=300000,
  				batch_size= 1024, ent_coef= 'auto', gamma= 0.9999, tau=0.02, train_freq= 64,  gradient_steps= 64,learning_starts= 10000,#target_update_interval=64,
  				use_sde= True, policy_kwargs= dict(log_std_init=-3, net_arch=[256,256,64]))#dict(pi=[256, 256], qf=[256, 256])))
		# from utils import plot_results,read_hyperparameters
		# hyperparams=read_hyperparameters('sac_cartpole_50')
		# model = SAC(env=env,**hyperparams)
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

