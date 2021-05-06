import sys
import os
STEPS_TO_TRAIN=150000
logdir='./logs/'
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
from utils import linear_schedule
from custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn import MlpPolicy
# from stable_baselines3.sac import MlpPolicy
from stable_baselines3 import DQN,SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from env_custom import CartPoleButter,CartPoleDiscreteHistory#,CartPoleContinous,CartPoleDiscreteHistory#,CartPoleDiscreteButter2
import argparse
EP_STEPS=800
Te=0.05


#TODO figure:  note as a function of steps.  3 curves:  1 DQN, 1 Q-learning with learning, 1Q-learning without learning
env = CartPoleButter(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMax=8.4706,resetMode='experimental',sparseReward=False)#,integrator='ode')#,integrator='rk4')
env = Monitor(env, filename=logdir+'simRaw')
model = DQN(MlpPolicy, env, learning_starts=10000, gamma=0.995,
            learning_rate=0.0003, exploration_final_eps=0.17787752200089602, exploration_initial_eps=0.17787752200089602,
            target_update_interval=1000, buffer_size=50000, n_episodes_rollout=1,train_freq=-1, gradient_steps=-1,
            verbose=0, batch_size=2024, policy_kwargs=dict(net_arch=[400, 300]))
#TODO temps d’apprentissage et note en fonction de l’amplitude du controle
tensionMax=[2.4, 3.5, 4.7, 5.9, 7.1, 8.2, 9.4, 12]
for tension in tensionMax:
    env = CartPoleButter(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMaxtension,resetMode='experimental',sparseReward=False)#,integrator='ode')#,integrator='rk4')
    env = Monitor(env, filename=logdir+'sim')
    model = DQN(MlpPolicy, env, learning_starts=10000, gamma=0.995,
                learning_rate=0.0003, exploration_final_eps=0.17787752200089602,
                exploration_initial_eps=0.17787752200089602,
                target_update_interval=1000, buffer_size=50000, n_episodes_rollout=1, train_freq=-1, gradient_steps=-1,
                verbose=0, batch_size=2024, policy_kwargs=dict(net_arch=[400, 300]))
#TODO standard deviation ofxas a function of the control amplitude in steadystate
#TODO standard deviation ofθas a function of the control amplitude in steadystate
#TODO temps d’apprentissage et note en fonction du coefficient de friction statique4 valeurs du coefficient:Ksc,virt= 0,0.1∗Ksc,manip,Ksc,manip,10∗Ksc,manipDiscussion
#TODO temps  d’apprentissage  et  note  en  fonction  du  coefficient  de  frictiondynamique
#TODO
#TODO


