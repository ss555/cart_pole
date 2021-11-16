'''
I3S lab:
By defaut the agent in trained and inference test is recorded at the end, results of an inference are recorded to .npz
If the WEIGHTS variable is not None, we try to load the selected weights to the model.

'''
import sys
import os
import torch
import re
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
from glob import glob
import numpy as np
import time
#Simulation parameters
Te = 0.05 #sampling time
EP_STEPS = 800 #num steps in an episode
STEPS_TO_TRAIN = 150000
PWM = 151 #PWM command to apply 0-255
INFERENCE_STEPS = 800 #steps to test the model

TRAIN = False#True #if true train, else only inf
x_threshold = 0.33 #limit on cart total: 33*2+5*2(hard)+4*2(soft) = 84 <84.5(rail)
MANUAL_SEED = 5


#paths to save monitor, models...
log_save = f'./weights/dqn50-real/pwm{PWM}'
Path(log_save).mkdir(parents=True, exist_ok=True)
WEIGHTS = None#f'./weights/dqn50-real/pwm{PWM}/dqn_rpi.zip'#None#f'./weights/dqn50-real/pwm{PWM}/dqn_rpi.zip'
REPLAY_BUFFER_WEIGHTS = None#f'./weights/dqn50-real/pwm{PWM}/dqn_rpi_buffer.pkl'  #None
logdir = f'./weights/dqn50-real/pwm{PWM}'
#initialisaiton of a socket and a gym env
pendulePy = PendulePy(wait=5, host='rpi5') #host:IP_ADRESS
env = CartPoleZmq(pendulePy=pendulePy, x_threshold=x_threshold, max_pwm = PWM)
torch.manual_seed(MANUAL_SEED)

TRAINING = False
INFERENCE_PATH = log_save

if __name__ == '__main__':
    try:

        if TRAIN:
            # callbacks
            # Use deterministic actions for evaluation and SAVE the best model
            # eval_callback = EvalCustomCallback(env, best_model_save_path=log_save + '/best.zip', log_path=log_save, eval_freq=10000, n_eval_episodes=1, deterministic=True, render=False)
            # callbackSave = SaveOnBestTrainingRewardCallback(log_dir=log_save, monitor_filename = log_save+'/training_exp_dqn.csv')
            env = Monitor(env, logdir)
            checkpoint = CheckPointEpisode(save_path=logdir, save_freq_ep=1)
            if WEIGHTS == None:
                hyperparams = read_hyperparameters('dqn_cartpole_50')
                model = DQN(env=env, seed = MANUAL_SEED, **hyperparams)
            else:  # transfer learning or inference #2.4 20&30 nothing

                try:
                    model = DQN.load(WEIGHTS, env=env, seed=MANUAL_SEED)
                except:
                    print(f'model not found on {WEIGHTS}')
                    sys.exit(1)
                model.learning_starts = 0
                model.exploration_initial_eps = model.exploration_final_eps
                if REPLAY_BUFFER_WEIGHTS is not None and WEIGHTS is not None:
                    model.load_replay_buffer(REPLAY_BUFFER_WEIGHTS)
                else:
                    print('warning: replay buffer trimmed')
            with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
                model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, checkpoint])
        else:
            modelsObsArr, modelActArr, modelRewArr = [],[],[]
            filenames = (sorted(glob(os.path.join(INFERENCE_PATH, "checkpoint*" + '.zip')), key=os.path.getmtime))
            for modelName in filenames:
                checkNum = re.findall('[0-9]+',modelName)
                if (int(checkNum[-1])-1)%10!=0:
                    continue
                print(f'loading {modelName}')
                s_time = time.time()
                model = DQN.load(modelName, env=env)
                print(f'loaded:{time.time()-s_time}')
                obs = env.reset()
                done = False
                obsArr, actArr, rewArr = [], [], []
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, done, _ = env.step(action)
                    obsArr.append(obs)
                    actArr.append(action)
                    rewArr.append(rewards)
                    if done:
                        #obs = env.reset() #put before?
                        break
                modelsObsArr.append(obsArr)
                modelActArr.append(actArr)
                modelRewArr.append(rewArr)
            np.savez(INFERENCE_PATH+'/inference_results.npz',modelsObsArr=modelsObsArr,modelActArr=modelActArr,modelRewArr=modelRewArr,filenames=filenames)
            pendulePy.sendCommand(0)
            sys.exit(0)
        model.env.MAX_STEPS_PER_EPISODE = INFERENCE_STEPS
        model.exploration_final_eps = 0.0
        model.exploration_initial_eps = 0.0
        obs = env.reset()
        for i in range(INFERENCE_STEPS):
            action, _states = model.predict(obs,deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if dones:
                env.reset()
                print('done reset')
    except Exception as e:
        print(e)
    finally:
        pendulePy.sendCommand(0)
        if TRAIN:
            model.save(logdir+f'pwm{PWM}/dqn_rpi.zip')
            model.save_replay_buffer(logdir+f'pwm{PWM}/dqn_rpi_buffer.pkl')

        copy_tree(logdir,'./../../../'+logdir+'-backup')