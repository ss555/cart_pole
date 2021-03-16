#tensorboard --logdir ./sac_cartpole_tensorboard/
import gym
import numpy as np
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC, DDPG,TD3, DQN
from env_custom import CartPoleCosSinDev,CartPoleCosSinTension,CartPoleCosSinTensionD
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import math
import time
from utils import plot
from matplotlib import pyplot as plt
Te = 5e-2
##Normaliser le modele
# env = CartPoleCosSinDev()#CartPoleCusBottom()CartPoleCosSin() #
# # env= CartPoleCosSinFriction()#CartPoleCusBottom()CartPoleCosSin() #
# env.MAX_STEPS_PER_EPISODE = 10000
# # Load the saved statistics
# env = DummyVecEnv([lambda: env])
# env = VecNormalize.load('envNorm.pkl', env)
# #  do not update them at test time
# # env=VecNormalize(env,norm_obs=True,norm_reward=False,clip_obs=10000,clip_reward=10000)
# env.training = False
# # reward normalization is not needed at test time
# env.norm_reward = False
# model = SAC.load("./logs/best_model", env=env)
# model = SAC.load("./logs/best_model_training.zip", env=env)
env = CartPoleCosSinTension()
# env = CartPoleCosSinDev()
actArr=[0.0]
timeArr=[0.0]
env.render()
start_time=time.time()
# mode='simplePi'
# mode='iniSpeed'
mode='oscillate'
if mode=='simplePi':
    obsIni = [env.reset(costheta=0, sintheta=1)]
    # obsIni=env.reset(costheta=-math.sqrt(2)/2, sintheta=math.sqrt(2)/2)
    print(obsIni)
    time.sleep(Te)
    obsArr = [obsIni]
    for i in range(1000):
        obs, rewards, dones, _ = env.step([0.0])
        # obs, rewards, dones, _ = env.step([action])
        # obsArr.append(math.atan2(obs[3], obs[2]))
        obsArr.append(obs)
        actArr.append(0.0)
        timeArr.append(time.time() - start_time)
        env.render()
        time.sleep(Te)
        if dones:
            print('reset: '+str(time.time()-start_time))
            env.reset()
            break
elif mode=='iniSpeed':
    obsArr = [env.reset(iniSpeed=0.4)]
    for i in range(100):
        action=[0.0]
        obs, rewards, dones, _ = env.step(action)
        # obs, rewards, dones, _ = env.step([action])
        # obsArr.append(math.atan2(obs[3], obs[2]))
        obsArr.append(obs)
        actArr.append(action[0])
        env.render()
        time.sleep(Te)
        if dones:
            print('reset: '+str(time.time()-start_time))
            env.reset()
            break
elif mode=='oscillate':
    obsArr = [env.reset()]
    # action=[100/180]
    action=[1]
    for i in range(1):
        for i in range(16):
            # action, _states = model.predict(obs, deterministic=True)
            env.render()
            obs, rewards, dones, _ = env.step(action)  # PWM - 100
            obsArr.append(obs)
            actArr.append(action[0])
            time.sleep(Te)
            timeArr.append(time.time() - start_time)
            # env.render()
            if dones:
                time.sleep(100)
                break
                env.reset()
        for i in range(2):
            # action, _states = model.predict(obs, deterministic=True)
            env.render()
            obs, rewards, dones, _ = env.step([0.0])  # PWM - 100
            obsArr.append(obs)
            actArr.append(action[0])
            time.sleep(Te)
            timeArr.append(time.time() - start_time)
            # env.render()
            if dones:
                break
                env.reset()
        for i in range(16):
            # action, _states = model.predict(obs, deterministic=True)
            env.render()
            obs, rewards, dones, _ = env.step([-action[0]])
            time.sleep(Te)
            obsArr.append(obs)
            actArr.append(action[0])
            timeArr.append(time.time() - start_time)
            if dones:
                break
                env.reset()
        for i in range(1):
            # action, _states = model.predict(obs, deterministic=True)
            env.render()
            obs, rewards, dones, _ = env.step([0.0])  # PWM - 100
            obsArr.append(obs)
            actArr.append(action[0])
            time.sleep(Te)
            timeArr.append(time.time() - start_time)
            # env.render()
            if dones:
                break
                env.reset()
plot(obsArr, timeArr,actArr)