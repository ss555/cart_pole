import sys
import os
import numpy as np
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule
from custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn import MlpPolicy
# from stable_baselines3.sac import MlpPolicy
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from env_custom import CartPoleButter
from utils import read_hyperparameters
import yaml
from pathlib import Path
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
STEPS_TO_TRAIN=100000
EP_STEPS=800
Te=0.05

#simulation results
qLearningVsDQN=False
# qLearningVsDQN=True
amplitudeVariationSim=False
# amplitudeVariationSim=True
DYNAMIC_FRICTION_SIM=False#True
STATIC_FRICTION_SIM=False
encNoiseVarSim=True

logdir='./EJPH/'
hyperparams=read_hyperparameters('dqn_cartpole_50')

if qLearningVsDQN:
    #DONE figure:  note as a function of steps.  3 curves:  1 DQN, 1 Q-learning with learning, 1Q-learning without learning
    Path('/EJPH/basic').mkdir(parents=True, exist_ok=True)
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, tensionMax=8.4706, resetMode='experimental', sparseReward=False,f_a=0,f_c=0,f_d=0, kPendViscous=0.0)#,integrator='ode')#,integrator='rk4')
    env = Monitor(env, filename=logdir+'basic/basic_simulation_')
    model = DQN(env=env, **hyperparams)
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    #TODO Q_learning
    plot_results(logdir+'basic')
#TODO temps d’apprentissage et note en fonction de l’amplitude du controle
if amplitudeVariationSim:
    Path('/EJPH/tension').mkdir(parents=True,exist_ok=True)
    tensionMax = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
    filenames=[]
    for tension in tensionMax:
        env = CartPoleButter(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMax=tension,resetMode='experimental',sparseReward=False)#,integrator='ode')#,integrator='rk4')
        filename=logdir + f'tension_sim_{tension}_V_'
        #filenames.append(filename)#NOT USED
        env = Monitor(env, filename=filename)
        model = DQN(env=env,**hyperparams)
        print(f'simulation for {tension} V')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    plot_results(logdir+'tension')

#TODO temps d’apprentissage et note en fonction du coefficient de friction statique 4 valeurs du coefficient:Ksc,virt= 0,0.1∗Ksc,manip,Ksc,manip,10∗Ksc,manipDiscussion
STATIC_FRICTION_CART=-0.902272006611719
STATIC_FRICTION_ARR=np.array([0,0.1,1,10])*STATIC_FRICTION_CART
if STATIC_FRICTION_SIM:
    Path('./EJPH/static-friction').mkdir(exist_ok=True)
    filenames=[]
    for frictionValue in STATIC_FRICTION_ARR:
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental',sparseReward=False,f_c=frictionValue)  # ,integrator='ode')#,integrator='rk4')
        filename = logdir + f'static-friction/static_friction_sim_{frictionValue}_'
        #filenames.append(filename)#NOT USED
        env = Monitor(env, filename=filename)
        model = DQN(env=env, **hyperparams)
        print(f'simulation with static friciton {frictionValue} coef')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    plot_results(logdir+'static-friction')
#TODO temps  d’apprentissage  et  note  en  fonction  du  coefficient  de  frictiondynamique
DYNAMIC_FRICTION_PENDULUM=0.11963736650935591
DYNAMIC_FRICTION_ARR=np.array([0,0.1,1,10])*DYNAMIC_FRICTION_PENDULUM
if DYNAMIC_FRICTION_SIM:
    Path('./EJPH/dynamic-friction').mkdir(exist_ok=True)
    filenames=[]
    for frictionValue in DYNAMIC_FRICTION_ARR:
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental', sparseReward=False,kPendViscous=frictionValue)  # ,integrator='ode')#,integrator='rk4')
        filename = logdir + f'dynamic-friction/dynamic_friction_sim_{frictionValue}_'
        #filenames.append(filename)#NOT USED
        env = Monitor(env, filename=filename)
        model = DQN(env=env, **hyperparams)
        print(f'simulation with dynamic friciton {frictionValue} coef')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    plot_results(logdir+'dynamic-friction')
#TODO encoder noise
NOISE_TABLE=np.array([0,0.01,0.05,0.1,0.15,0.5,1,5,10])*np.pi/180
'''
Noise effects (DQN)add noise to position: gaussian,σ(θ) = 0,0.01,0.05,0.10.15,0.5,1,5,10◦.we correlate simultaneously to the noise in   ̇σ,σ( ̇θ) =σ(θ)/∆twith ∆t= 50ms.Figure :  temps d’apprentissage en fonction deσθ
'''
if encNoiseVarSim:
    Path('./EJPH/encoder-noise').mkdir(exist_ok=True)
    filenames=[]
    for encNoise in NOISE_TABLE:
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental', sparseReward=False,Km=encNoise)  # ,integrator='ode')#,integrator='rk4')
        filename = logdir + f'encoder-noise/enc_noise_sim_{encNoise}_rad_'
        #filenames.append(filename)#NOT USED
        env = Monitor(env, filename=filename)
        model = DQN(env=env, **hyperparams)
        print(f'simulation with noise {encNoise*180/np.pi} degree')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    plot_results(logdir+'encoder-noise')
#TODO effect of initialisation
RESET_EFFECT=False
if RESET_EFFECT:
    Path('./EJPH/experimental-vs-random').mkdir(exist_ok=True)
    filenames=[]
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental', sparseReward=False)  # ,integrator='ode')#,integrator='rk4')
    filename = logdir + f'experimental-vs-random/experimental'
    #filenames.append(filename)#NOT USED
    env = Monitor(env, filename=filename)
    model = DQN(env=env, **hyperparams)
    print(f'simulation with experimental reset')
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    print(f'simulation with random reset')
    env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True, resetMode='random', sparseReward=False)
    filename = logdir + f'experimental-vs-random/random'
    env = Monitor(env, filename=filename)
    model = DQN(env=env, **hyperparams)
    plot_results(logdir+'experimental-vs-random')
#TODO standard deviation of xas a function of the control amplitude in steadystate
#TODO standard deviation of θas a function of the control amplitude in steadystate
#TODO bruit sur action [0, 0.1%, 1%, 10%]


