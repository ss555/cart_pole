import sys
import os
sys.path.append(os.path.abspath('/'))
from src.utils import linear_schedule
from src.custom_callbacks import plot_results
from src.env_wrappers import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from src.custom_callbacks import EvalCustomCallback, EvalX_ThetaMetric
from src.custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from src.env_custom import CartPoleRK4
import argparse
from src.utils import read_hyperparameters
from pathlib import Path

Te=0.05
EP_STEPS=800
STEPS_TO_TRAIN=90000
LOAD_MODEL_PATH=None#"./logs/best_model"
LOAD_BUFFER_PATH=None#"dqn_pi_swingup_bufferN"
logdir='./logs/'
checkpoint = CheckpointCallback(save_freq=1000, save_path=logdir)
env = CartPoleRK4(Te=Te, N_STEPS=EP_STEPS,discreteActions=True, f_c=-11.66390864012042, resetMode='experimental',sparseReward=False,Km=0.0,n=1)#,integrator='ode')#,integrator='rk4')
env = Monitor(env, filename=logdir+'basic_simulation_')
envEvaluation = CartPoleRK4(Te=Te, N_STEPS=EP_STEPS,discreteActions=True, f_c=-2*11.66390864012042,resetMode='random',sparseReward=False,Km=0.0,n=1)#,integrator='ode')#,integrator='rk4')
#integrator='rk4',
NORMALISE=False
if NORMALISE:
    ## Automatically normalize the input features and reward
    env1 = DummyVecEnv([lambda: env])
    # env = VecNormalize.load('envNorm.pkl', env)
    env = VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=1000, clip_reward=1000)
    print('using normalised env')
    env.training = True
else:
    envEval=env


log_save='./weights/dqn50-sim'
Path(log_save).mkdir(exist_ok=True)
#callbacks
# Use deterministic actions for evaluation and SAVE the best model
# eval_callback = EvalCustomCallback(envEvaluation, best_model_save_path=log_save, log_path=logdir+'/evals', eval_freq=15000, n_eval_episodes=30,deterministic=True, render=False)
eval_callback = EvalX_ThetaMetric(envEvaluation, best_model_save_path=log_save,N_BINS=4, X_THRESHOLD=0.2,log_path=logdir+'/evals', eval_freq=5000, verbose=1)
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
hyperparams = read_hyperparameters('dqn_cartpole_50')
model = DQN(env=env, **hyperparams)


# model for pendulum starting from bottom
with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
    model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback, checkpoint])
    if NORMALISE:
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
        env.save(logdir + 'envNorm.pkl')
    plot_results(logdir)
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()


model.save("deepq_cartpole")
model.save_replay_buffer('replayBufferDQN')
# WHEN NORMALISING
if NORMALISE:
    env.save(logdir + 'envNorm.pkl')
plot_results(logdir)