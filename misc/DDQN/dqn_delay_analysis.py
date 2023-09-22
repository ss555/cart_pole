import sys
import os
sys.path.append(os.path.abspath('/'))
from src.utils import linear_schedule
from src.custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from src.custom_callbacks import EvalCustomCallback
from src.custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from src.env_custom import CartPoleButterHistoryDelay
import argparse
from src.utils import read_hyperparameters
from pathlib import Path

Te=0.05
EP_STEPS=800
STEPS_TO_TRAIN=90000
LOAD_MODEL_PATH=None  #"./logs/best_model"
LOAD_BUFFER_PATH=None #"dqn_pi_swingup_bufferN"
logdir = './logs/'
VOLTAGE = 12 #8.4706
# env = CartPoleDiscreteHistory()
env = CartPoleButterHistoryDelay(tensionMax=VOLTAGE, n=2, FILTER=False, kS=5, delay=1)
# env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS,discreteActions=True,tensionMax=VOLTAGE, resetMode='experimental',sparseReward=False,Km=0.0,n=1,FILTER=True)
env = Monitor(env, filename=logdir+'basic_simulation_')
# envEvaluation = CartPoleDiscreteHistory()
envEvaluation = CartPoleButterHistoryDelay(tensionMax=VOLTAGE, n=2, FILTER=False, kS=5, delay=1)

# envEvaluation = CartPoleButter(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMax=VOLTAGE, resetMode='experimental',sparseReward=False,Km=0.0,n=1,FILTER=True)





log_save='./weights/dqn50-sim'
Path(log_save).mkdir(exist_ok=True)
#callbacks
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCustomCallback(envEvaluation, best_model_save_path=log_save, log_path=logdir+'/evals', eval_freq=15000, n_eval_episodes=5,deterministic=True, render=False)
hyperparams = read_hyperparameters('dqn_cartpole_50')
model       = DQN(env=env,seed=5,**hyperparams)
callbackSave = SaveOnBestTrainingRewardCallback(log_dir=log_save, monitor_filename=logdir+'basic_simulation_monitor.csv')

try:
    #model for pendulum starting from bottom
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback, callbackSave])
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
    plot_results(logdir)