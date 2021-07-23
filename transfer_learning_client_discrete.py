from glob import glob
from tcp_envV2 import CartPoleCosSinRpiDiscrete3
from stable_baselines3 import DQN
from utils import linear_schedule, plot
from stable_baselines3.common.monitor import Monitor
from custom_callbacks import ProgressBarManager, CheckPointEpisode
import socket
import time
import os
from utils import read_hyperparameters
import numpy as np
from custom_callbacks import plot_results
'''
three modes: TRAIN(train model from scratch or LOAD_MODEL_PATH), 
INFERENCE(many models from INFERENCE_PATH), ONE_TIME_INFERENCE(1model of LOAD_MODEL_PATH!!)
'''
mode='TRAIN'#INFERENCE #ONE_TIME_INFERENCE
HOST = '134.59.131.77'
LOAD_MODEL_PATH=None#"./logs/best_model"
LOAD_BUFFER_PATH=None#"dqn_pi_swingup_bufferN"
INFERENCE_PATH='./EJPH/real-cartpole/dqn'
PORT = 65432
TENSION = 12
logdir=f'./weights/dqn{TENSION}V/'
os.makedirs(logdir,exist_ok=True)
# Use deterministic actions for evaluation and SAVE the best model
checkpoint = CheckPointEpisode(save_path=logdir)
STEPS_TO_TRAIN = 150000
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
            model.gradient_steps=-1
            model.train_freq=-1
            model.n_episodes_rollout=1
            model.exploration_final_eps=0.05
        if LOAD_BUFFER_PATH!=None:
            model.load_replay_buffer(LOAD_BUFFER_PATH)
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
            filenames = (sorted(glob(os.path.join(INFERENCE_PATH, "*" + '.zip')), key=os.path.getmtime))
            obs = env.reset()
            for modelName in filenames:
                print(f'loading {modelName}')
                model = DQN.load(modelName, env=env)
                done = False
                obsArr, actArr, rewArr = [], [], []
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, done, _ = env.step(action)
                    obsArr.append(obs)
                    actArr.append(action)
                    rewArr.append(rewards)
                    if done:
                        obs = env.reset()
                        break
                modelsObsArr.append(obsArr)
                modelActArr.append(actArr)
                modelRewArr.append(rewArr)
            np.savez('./EJPH/real-cartpole/dqn/inference_results.npz',modelsObsArr=modelsObsArr,modelActArr=modelActArr,modelRewArr=modelRewArr,filenames=filenames)
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