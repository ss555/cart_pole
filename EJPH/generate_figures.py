import sys
import os
import numpy as np
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule, plot_results
from custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn import MlpPolicy
# from stable_baselines3.sac import MlpPolicy
from stable_baselines3 import DQN,SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from env_custom import CartPoleButter
from utils import plot_results
import argparse
import yaml
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
STEPS_TO_TRAIN=100000
EP_STEPS=800
Te=0.05
#simulation results
qLearningVsDQN=False
# qLearningVsDQN=True
amplitudeVariationSim=True
# amplitudeVariationSim=False

logdir='./EJPH/'

with open("parameters.yml", "r") as f:
    hyperparams_dict = yaml.safe_load(f)
    hyperparams=hyperparams_dict['dqn_sim50']
    if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"],list):
        hyperparams["train_freq"]=tuple(hyperparams["train_freq"])
        print('parameters loaded')
    # Convert to python object if needed
    if isinstance(hyperparams["policy_kwargs"], str):
        hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
    # Overwrite hyperparams if needed
    # hyperparams.update(self.custom_hyperparams)
if qLearningVsDQN:
    #TODO figure:  note as a function of steps.  3 curves:  1 DQN, 1 Q-learning with learning, 1Q-learning without learning
    env = CartPoleButter(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMax=8.4706,resetMode='experimental',sparseReward=False,f_a=0,f_c=0,f_d=0)#,integrator='ode')#,integrator='rk4')
    env = Monitor(env, filename=logdir+'basic_simulation_')
    model = DQN(env=env,**hyperparams)
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN,callback=[cus_callback])
    #Q_learning
    plot_results('./basic_simulation_monitor.csv')
#TODO temps d’apprentissage et note en fonction de l’amplitude du controle
if amplitudeVariationSim:
    tensionMax = [2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
    filenames=[]
    for tension in tensionMax:
        env = CartPoleButter(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMax=tension,resetMode='experimental',sparseReward=False)#,integrator='ode')#,integrator='rk4')
        filename=logdir + f'tension_sim_{tension}_V_'
        filenames.append(filename)
        env = Monitor(env, filename=filename)
        model = DQN(env=env,**hyperparams)
        print(f'simulation for {tension} V')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    plot_results(filename)

#TODO standard deviation of xas a function of the control amplitude in steadystate
#TODO standard deviation of θas a function of the control amplitude in steadystate
#TODO temps d’apprentissage et note en fonction du coefficient de friction statique 4 valeurs du coefficient:Ksc,virt= 0,0.1∗Ksc,manip,Ksc,manip,10∗Ksc,manipDiscussion
STATIC_FRICTION_CART=-0.902272006611719
STATIC_FRICTION_ARR=[0,0.1,1,10]*STATIC_FRICTION_CART
#TODO temps  d’apprentissage  et  note  en  fonction  du  coefficient  de  frictiondynamique
DYNAMIC_FRICTION_PENDULUM=-21
DYNAMIC_FRICTION_ARR=[0,0.1,1,10]*DYNAMIC_FRICTION_PENDULUM
DYNAMIC_FRICTION_SIM=True
if DYNAMIC_FRICTION_SIM:
    for frictionValue in DYNAMIC_FRICTION_ARR:
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental', sparseReward=False,f_a=frictionValue)  # ,integrator='ode')#,integrator='rk4')
        filename = logdir + f'dynamic_friction_sim_{frictionValue}_'
        filenames.append(filename)
        env = Monitor(env, filename=filename)
        model = DQN(env=env, **hyperparams)
        print(f'simulation with dynamic friciton {frictionValue} coef')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    plot_results(filename)
#TODO encoder noise
NOISE_TABLE=[0,0.01,0.05,0.10.15,0.5,1,5,10]/np.pi
'''
Noise effects (DQN)add noise to position: gaussian,σ(θ) = 0,0.01,0.05,0.10.15,0.5,1,5,10◦.we correlate simultaneously to the noise in   ̇σ,σ( ̇θ) =σ(θ)/∆twith ∆t= 50ms.Figure :  temps d’apprentissage en fonction deσθ
'''
encNoiseVarSim=True
if encNoiseVarSim:
    for encNoise in NOISE_TABLE:
        env = CartPoleButter(Te=Te, N_STEPS=EP_STEPS, discreteActions=True,resetMode='experimental', sparseReward=False)  # ,integrator='ode')#,integrator='rk4')
        filename = logdir + f'enc_noise_sim_{encNoise}_rad_'
        filenames.append(filename)
        env = Monitor(env, filename=filename)
        model = DQN(env=env, **hyperparams)
        print(f'simulation with noise {encNoise*np.pi} degree')
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback])
    plot_results(filename)
#TODO


