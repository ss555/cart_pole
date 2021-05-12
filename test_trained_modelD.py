#tensorboard --logdir ./sac_cartpole_tensorboard/
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import DQN
from env_custom import CartPoleButter#,CartPoleDiscreteHistory #CartPoleCosSinTensionD,CartPoleCosSinTensionD3,
from sb3_contrib import QRDQN
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from matplotlib import pyplot as plt
from utils import plot, plot_line
import numpy as np

Te = 5e-2
#CartPoleCosSinHistory() #CartPoleCusBottom()CartPoleCosSin() #
env = CartPoleButter(Te=0.02,N_STEPS=1500,discreteActions=True,resetMode='experimental',sparseReward=False)
# Load the saved statistics
env.MAX_STEPS_PER_EPISODE = 800
model = DQN.load("./logs/best_model", env=env)
env.randomReset=False
obs = env.reset(xIni=0)
env.render()
observations=[]
start_time=time.time()
obsArr=[obs]
actArr=[0.0]
timeArr=[0.0]
model.exploration_initial_eps=0
model.exploration_final_eps=0
rewardsArr=[]
for i in range(1600):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ = env.step(action)
    rewardsArr.append(rewards)
    obsArr.append(obs)
    actArr.append(action)
    timeArr.append(time.time()-start_time)
    # time.sleep(Te)
    env.render()
    if dones:
        break
        #env.reset()
plot(obsArr,timeArr,actArr,plotlyUse=False)
print(f'average reward: {np.mean(rewardsArr)}')
# f_a = 13.52
# f_b = -0.72
# f_c = -0.68
# dv = -a * vs[i] + b * u + c * np.sign(vs[i]), donc u=(dv+avs[i]-c*np.sign(vs[i]))/b
# staticV= c/b=1=21PWM(0,08 en action), dynamique K1: 18.8, F
#u=-a/b,donc fk