import sys
import os
sys.path.append(os.path.abspath('./'))
from utils import linear_schedule
from custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import EvalCustomCallback
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from env_custom import CartPoleButter, CartPoleDiscreteHistory, CartPoleButterHistoryDelay#, CartPoleButterHistory
import argparse
from utils import read_hyperparameters
from pathlib import Path

Te=0.05
EP_STEPS=800
STEPS_TO_TRAIN=90000
SAVE_MODEL_PATH = "deepq_cartpole.zip"
SAVE_BUFFER_PATH = 'replayBufferDQN.pkl'
LOAD_MODEL_PATH = None#SAVE_MODEL_PATH#'./weights/dqn50-sim/best_model_training.zip'#SAVE_MODEL_PATH#None#'./weights/dqn50-sim/best_model_training.zip'
LOAD_BUFFER_PATH = None#SAVE_BUFFER_PATH#"dqn_pi_swingup_bufferN"
logdir = './logs/'
VOLTAGE = 12 #8.4706
env = CartPoleButterHistoryDelay(Te=Te, N_STEPS=EP_STEPS,discreteActions=True,tensionMax=VOLTAGE, resetMode='experimental',sparseReward=False,Km=0.0,n=5,FILTER=False)
env = Monitor(env, filename=logdir+'basic_simulation_')

envEvaluation = CartPoleButterHistoryDelay(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMax=VOLTAGE, resetMode='experimental',sparseReward=False,Km=0.0,n=5,FILTER=False)
# envEvaluation = CartPoleButter(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMax=VOLTAGE, resetMode='experimental',sparseReward=False,Km=0.0,n=1,FILTER=True)
NORMALISE = False

if NORMALISE:
    ## Automatically normalize the input features and reward
    env1 = DummyVecEnv([lambda: env])
    env = VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=1000, clip_reward=1000)
    print('using normalised env')
    env.training = True
else:
    envEval = env




log_save='./weights/other'
# log_save='./weights/dqn50-sim'
Path(log_save).mkdir(exist_ok=True)
#callbacks
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCustomCallback(envEvaluation, best_model_save_path=log_save, log_path=logdir+'/evals', eval_freq=15000, n_eval_episodes=30,deterministic=True, render=False)

if LOAD_MODEL_PATH is None:
    hyperparams = read_hyperparameters('dqn_cartpole_50')
    model = DQN(env=env,seed=5,**hyperparams)
    #we load buffer with the model
    if LOAD_BUFFER_PATH != None:
        model.load_replay_buffer(LOAD_BUFFER_PATH)
        print('loaded replay buffer')
else:
    print('preloading weights')
    model = DQN.load(LOAD_MODEL_PATH, env=env)
    model.learning_starts = 0
    # model.exploration_rate = 0
    model.exploration_initial_eps = model.exploration_final_eps


callbackSave = SaveOnBestTrainingRewardCallback(log_dir=log_save, monitor_filename=logdir+'basic_simulation_monitor.csv')

try:
    # model for pendulum starting from bottom
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback, callbackSave])
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
    model.save(SAVE_MODEL_PATH)
    model.save_replay_buffer(SAVE_BUFFER_PATH)
    # WHEN NORMALISING
    if NORMALISE:
        env.save(logdir + 'envNorm.pkl')
    plot_results(logdir)