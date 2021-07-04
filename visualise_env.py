#tensorboard --logdir ./sac_cartpole_tensorboard/
import gym
from env_custom import CartPoleButter
import time
from utils import plot
from matplotlib import pyplot as plt
import cartpole
Te = 5e-2
N = 1
# env = gym.make('CartPoleImageC-v0')
env = CartPoleButter(discreteActions=False)#Te=Te,n=N,integrator='semi-euler',resetMode='experimental')
actArr=[0.0]
timeArr=[0.0]
env.reset(xIni=0)
env.render()
start_time=time.time()
mode='simplePi'
# mode='iniSpeed'
# mode='oscillate'
DISCRETE=type(env.action_space)==gym.spaces.discrete.Discrete
if mode=='simplePi':
    obsIni = [env.reset(costheta=0, sintheta=1,xIni=0)]
    # obsIni=env.reset(costheta=-math.sqrt(2)/2, sintheta=math.sqrt(2)/2)
    print(obsIni)
    # time.sleep(Te/N)
    obsArr = [obsIni]
    for i in range(2000):
        if DISCRETE:
            obs, rewards, dones, _ = env.step(1)#FOR DISCRETE
        else:
            obs, rewards, dones, _ = env.step([0])
        # obs, rewards, dones, _ = env.step(0)
        # obs, rewards, dones, _ = env.step([action])
        # obsArr.append(math.atan2(obs[3], obs[2]))
        obsArr.append(obs)
        actArr.append(0.0)
        timeArr.append(time.time() - start_time)
        env.render()
        # time.sleep(Te)
        if dones:
            print('reset: '+str(time.time()-start_time))
            env.reset()
            break
elif mode=='iniSpeed':
    obsArr = [env.reset(iniSpeed=0.5)]
    for i in range(100):
        action=[0.0]
        if DISCRETE:
            obs, rewards, dones, _ = env.step(1)  # FOR DISCRETE
        else:
            obs, rewards, dones, _ = env.step(action)
        obsArr.append(obs)
        actArr.append(action[0])
        env.render()
        time.sleep(Te)
        if dones:
            print('reset: '+str(time.time()-start_time))
            env.reset()
            break
elif mode == 'oscillate':
    obsArr = [env.reset()]
    action=[1.0]
    for i in range(10):
        for i in range(8):
            # action, _states = model.predict(obs, deterministic=True)
            env.render()
            obs, rewards, dones, _ = env.step(action)  # PWM - 100
            obsArr.append(obs)
            actArr.append(action[0])
            time.sleep(Te)
            timeArr.append(time.time() - start_time)
            # env.render()
            if dones:
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
        for i in range(8):
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
plot(obsArr, timeArr,actArr,plotlyUse=True)