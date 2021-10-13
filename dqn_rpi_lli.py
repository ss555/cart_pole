'''
I3S lab:
By defaut the agent in trained and inference test is recorded at the end, results of an inference are recorded to .npz
If the WEIGHTS variable is not None, we try to load the selected weights to the model.

'''
import sys
import os
import torch
from distutils.dir_util import copy_tree
sys.path.append(os.path.abspath('./..'))
sys.path.append(os.path.abspath('./'))
from custom_callbacks import plot_results
from env_wrappers import Monitor
from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import EvalCustomCallback, CheckPointEpisode
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from tcp_envV2 import CartPoleZmq
from utils import read_hyperparameters
from pathlib import Path
from pendule_pi import PendulePy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
#Simulation parameters
Te=0.05 #sampling time
EP_STEPS=800 #num steps in an episode
STEPS_TO_TRAIN=150000
PWM = 255 #PWM command to apply 0-255
INFERENCE_STEPS = 2000 #steps to test the model
MANUAL_SEED=0 #seed to fix on the torch

TRAIN = True#True #if true train, else only inf
x_threshold = 0.33 #limit on cart total: 33*2+5*2(hard)+4*2(soft) = 84 <84.5(rail)
MANUAL_SEED = 5


#paths to save monitor, models...
log_save = f'./weights/dqn50-real/pwm{PWM}'
Path(log_save).mkdir(parents=True, exist_ok=True)
WEIGHTS = f'./weights/dqn50-real/pwm{PWM}/dqn_rpi.zip'#None#f'./weights/dqn50-real/pwm{PWM}/dqn_rpi.zip'
REPLAY_BUFFER_WEIGHTS = f'./weights/dqn50-real/pwm{PWM}/dqn_rpi_buffer.pkl'  #None
logdir = f'./weights/dqn50-real/pwm{PWM}'
#initialisaiton of a socket and a gym env
pendulePy = PendulePy(wait=5, host='rpi5') #host:IP_ADRESS
env0 = CartPoleZmq(pendulePy=pendulePy, x_threshold=x_threshold, max_pwm = PWM)
torch.manual_seed(MANUAL_SEED)
env = Monitor(env0,logdir)
TRAINING = False
if __name__ == '__main__':
    try:
        # callbacks
        # Use deterministic actions for evaluation and SAVE the best model
        #eval_callback = EvalCustomCallback(env, best_model_save_path=log_save + '/best.zip', log_path=log_save, eval_freq=10000, n_eval_episodes=1, deterministic=True, render=False)
        #callbackSave = SaveOnBestTrainingRewardCallback(log_dir=log_save, monitor_filename = log_save+'/training_exp_dqn.csv')
        checkpoint = CheckPointEpisode(save_path=logdir, episodes_init=96)
        if WEIGHTS == None:
            hyperparams = read_hyperparameters('dqn_cartpole_50')
            model = DQN(env=env, **hyperparams)
        else:#transfer learning or inference

            try:
                model = DQN.load(WEIGHTS, env=env, seed=MANUAL_SEED)
                model.learning_starts = 0
                model.exploration_initial_eps = model.exploration_final_eps
                model.exploration_final_eps = 0.0
                model.exploration_initial_eps = 0.0
            except:
                print(f'model not found on {WEIGHTS}')

            if REPLAY_BUFFER_WEIGHTS is not None and WEIGHTS is not None:
                model.load_replay_buffer(REPLAY_BUFFER_WEIGHTS)
        if TRAIN:
            TRAINING = True
            with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
                model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, checkpoint])
        model.env.MAX_STEPS_PER_EPISODE = INFERENCE_STEPS
        model.env.MAX_STEPS_PER_EPISODE = EP_STEPS
        model.exploration_final_eps = 0.0
        model.exploration_initial_eps = 0.0
        obs = env.reset()
        for i in range(INFERENCE_STEPS):
            action, _states = model.predict(obs,deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if dones:
                env.reset()
                print('done reset')

    finally:
        pendulePy.sendCommand(0)
        if TRAINING:
            model.save(logdir+f'pwm{PWM}/dqn_rpi.zip')
            model.save_replay_buffer(logdir+f'pwm{PWM}/dqn_rpi_buffer.pkl')
            plot_results(log_save)

        copy_tree(logdir,'./../'+logdir)