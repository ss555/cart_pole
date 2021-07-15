import sys
import os
sys.path.append(os.path.abspath('./'))
from utils import linear_schedule
from custom_callbacks import plot_results,CheckPointEpisode
# from env_wrappers import Monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback, EvalCustomCallback, EvalThetaDotMetric
from env_custom import CartPoleButter, CartPoleDebug, CartPoleDiscreteHistory#,CartPoleContinous,CartPoleDiscreteHistory#,CartPoleDiscreteButter2
import argparse
import numpy as np
from utils import read_hyperparameters
from pathlib import Path
Te=0.05
EP_STEPS=800
STEPS_TO_TRAIN=60000
LOAD_MODEL_PATH=None#"./logs/best_model"
LOAD_BUFFER_PATH=None#"dqn_pi_swingup_bufferN"
logdir = './logs/'
checkpoint = CheckPointEpisode(save_path=logdir+'checkpoints')
env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=7.1, Km=np.pi/360,
                              resetMode='experimental', sparseReward=False)
env = Monitor(env, filename=logdir+'basic_simulation_')
# env = DummyVecEnv([lambda: env])
envEvaluation = env#CartPoleButter(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMax=8.4706,resetMode='experimental')#,integrator='ode')#,integrator='rk4')
NORMALISE = False
if NORMALISE:
    ## Automatically normalize the input features and reward
    env1 = DummyVecEnv([lambda: env])
    # env = VecNormalize.load('envNorm.pkl', env)
    env = VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=1000, clip_reward=1000)
    print('using normalised env')
    env.training = True
else:
    envEval = env




log_save='./weights/dqn50'
Path(log_save).mkdir(exist_ok=True)
#callbacks
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalThetaDotMetric(envEvaluation, log_path=logdir, eval_freq=6000, deterministic=True,verbose=1)
# eval_callback = EvalCustomCallback(envEvaluation, best_model_save_path=log_save, log_path=logdir+'/evals', eval_freq=STEPS_TO_TRAIN/3, n_eval_episodes=30,deterministic=True, render=False)
hyperparams = read_hyperparameters('dqn_50')
model = DQN(env=env,**hyperparams)
callbackSave = SaveOnBestTrainingRewardCallback(log_dir=log_save, monitor_filename=logdir+'basic_simulation_monitor.csv')

try:
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

finally:
    model.save("deepq_cartpole")
    model.save_replay_buffer('replayBufferDQN')
    # WHEN NORMALISING
    if NORMALISE:
        env.save(logdir + 'envNorm.pkl')
    plot_results(logdir)