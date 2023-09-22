#tensorboard --logdir ./sac_cartpole_tensorboard/
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import DQN
from src.env_custom import CartPoleButter#,CartPoleDiscreteHistory #CartPoleCosSinTensionD,CartPoleCosSinTensionD3,
from sb3_contrib import QRDQN
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from matplotlib import pyplot as plt
from src.utils import plot, plot_line
import numpy as np
Te = 5e-2
#CartPoleCosSinHistory() #CartPoleCusBottom()CartPoleCosSin() #
STEPS = 5000
env = CartPoleButter(Te=Te,tensionMax=8.4706,resetMode='experimental',sparseReward=False,Km=0.0,n=1)#,integrator='ode')#,integrator='rk4')
# Load the saved statistics
env.MAX_STEPS_PER_EPISODE = STEPS
model = DQN.load('./weights/dqn50-sim/best_model.zip', env=env)
# model = DQN.load("./logs/best_model", env=env)
env.randomReset = False
obs = env.reset(xIni=0)
# env.render()
observations = []
start_time = time.time()
obsArr=[obs]
actArr=[0.0]
timeArr=[0.0]
model.exploration_initial_eps=0
model.exploration_final_eps=0
rewardsArr=[]
for i in range(STEPS):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ = env.step(action)
    rewardsArr.append(rewards)
    obsArr.append(obs)
    actArr.append(action)
    timeArr.append(time.time()-start_time)
    # time.sleep(Te)
    # env.render()
    if dones:
        print(f'interrupted on {i}')
        break
        #env.reset()
obsArr=np.array(obsArr)
plot(obsArr,timeArr,actArr,plotlyUse=False)
indexStart=1000
print(f'average reward: {np.mean(rewardsArr)}')
print(f'mean x : {np.mean(obsArr[indexStart:,0])}')
print(f'std on x: {np.std(obsArr[indexStart:, 0])}')
print(f'mean theta : {np.mean(np.arctan2(obsArr[indexStart:, 3],obsArr[indexStart:, 2]))}')
print(f'std on theta: {np.std(obsArr[indexStart:, 0])}')

# f_a = 13.52
# f_b = -0.72
# f_c = -0.68
# dv = -a * vs[i] + b * u + c * np.sign(vs[i]), donc u=(dv+avs[i]-c*np.sign(vs[i]))/b
# staticV= c/b=1=21PWM(0,08 en action), dynamique K1: 18.8, F
#u=-a/b,donc fk