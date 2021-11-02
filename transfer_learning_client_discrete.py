from glob import glob
from tcp_envV2 import CartPoleCosSinRpiDiscrete3
from stable_baselines3 import DQN
from utils import plot
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager, CheckPointEpisode
import socket
import time
import os
from utils import read_hyperparameters
import numpy as np
from custom_callbacks import plot_results
'''
three modes: TRAIN (train model from scratch or LOAD_MODEL_PATH), 
INFERENCE (many models from INFERENCE_PATH), ONE_TIME_INFERENCE (1model of LOAD_MODEL_PATH!!) '''
mode = 'INFERENCE'  #'INFERENCE' #
HOST = '134.59.131.77'
LOAD_MODEL_PATH = None #'./weights/dqn2.4V/cartpole_pi_dqnN'#'./EJPH/real-cartpole/dqn/end/cartpole_pi_dqnN'#"./logs/best_model"
LOAD_BUFFER_PATH = None #'./weights/dqn2.4V/dqn_pi_swingup_bufferN.pkl'#'./EJPH/real-cartpole/dqn/dqn_pi_swingup_bufferN.pkl'#"dqn_pi_swingup_bufferN"

PORT = 65432
TENSION = 12 #3.5 #75pwm 2.4 45PWM
logdir = f'./weights/dqn12V/continue'
#SPECIFY INFERENCE PATH IF mode = 'INFERENCE'
INFERENCE_PATH = logdir#'./EJPH/real-cartpole/dqn'
os.makedirs(logdir, exist_ok=True)
checkpoint = CheckPointEpisode(save_path=logdir, episodes_init=0)
STEPS_TO_TRAIN = 60000
#lr schedule
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        env = CartPoleCosSinRpiDiscrete3(pi_conn=s)
        env0 = Monitor(env, logdir)
        # eval_callback = EvalCallback(env, best_model_save_path=logdir, n_eval_episodes=2,
        #                              log_path=logdir, eval_freq=10000,
        #                              deterministic=True, render=False)
        hyperparams = read_hyperparameters('dqn_cartpole_50')
        model = DQN(env=env0, seed=0, **hyperparams)
        if LOAD_MODEL_PATH!=None:
            model = DQN.load(LOAD_MODEL_PATH, env=env0)
            model.learning_starts=0
            print('loaded model')
        if LOAD_BUFFER_PATH!=None:
            model.load_replay_buffer(LOAD_BUFFER_PATH)
            print('loaded replay buffer')
        if mode=='TRAIN':
            with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
                model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, checkpoint])
        elif mode=='ONE_TIME_INFERENCE':
            obs = env.reset()
            obsArr=[obs]
            actArr=[0.0]
            timeArr=[0.0]
            start_time = time.time()
            for i in range(1000):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = env.step(action)
                obsArr.append(obs)
                actArr.append(action)
                timeArr.append(time.time()-start_time)
                if dones:
                    env.reset()
            env.reset()
        elif mode == 'INFERENCE':
            modelsObsArr, modelActArr, modelRewArr = [],[],[]
            filenames = (sorted(glob(os.path.join(INFERENCE_PATH, "checkpoint*" + '.zip')), key=os.path.getmtime))
            for modelName in filenames:
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
finally:
    if mode=='TRAIN':
        model.save(logdir+"/cartpole_pi_dqnN")
        model.save_replay_buffer(logdir+"/dqn_pi_swingup_bufferN")
    s.close()
    try: #if training was interrupted
        plot_results(logdir)
        plot(obsArr, timeArr, actArr)
    finally:
        pass