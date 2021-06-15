import sys
import os
import torch
sys.path.append(os.path.abspath('./'))
from custom_callbacks import plot_results
from env_wrappers import Monitor
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import EvalCustomCallback
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from tcp_envV2 import CartPoleZmq
import argparse
from utils import read_hyperparameters
from pathlib import Path
from pendule_pi import PendulePy

Te=0.05
EP_STEPS=800
STEPS_TO_TRAIN=200000
LOAD_MODEL_PATH=None#"./logs/best_model"
LOAD_BUFFER_PATH=None#"dqn_pi_swingup_bufferN"
logdir='./logs/'
Path(logdir).mkdir(parents=True, exist_ok=True)
log_save='./weights/dqn50-sim'
Path(log_save).mkdir(parents=True, exist_ok=True)
pendulePy = PendulePy(wait=5,host='rpi5')
env0 = CartPoleZmq(pendulePy=pendulePy,max_pwm=150)
env = Monitor(env0, filename=log_save)
MANUAL_SEED=0
torch.manual_seed(MANUAL_SEED)
TRAIN=True

# envEval = CartPoleZmq(pendulePy=pendulePy,MAX_STEPS_PER_EPISODE=2000)
#callbacks
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCustomCallback(env0, best_model_save_path=log_save, log_path=logdir, eval_freq=15000, n_eval_episodes=1,deterministic=True, render=False)
callbackSave = SaveOnBestTrainingRewardCallback(log_dir=logdir)


# if __name__==
try:
    if TRAIN:
        hyperparams = read_hyperparameters('dqn_cartpole_50')
        model = DQN(env=env, **hyperparams)
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback, callbackSave])
    else:#inference
        model = DQN.load('deepq_cartpole.zip', env=env0)
        model.learning_starts = 0

    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs,deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            env.reset()
            print('done reset')

finally:
    pendulePy.sendCommand(0)
    if TRAIN:
        model.save("deepq_cartpole")
        model.save_replay_buffer('replayBufferDQN')
        plot_results(logdir)