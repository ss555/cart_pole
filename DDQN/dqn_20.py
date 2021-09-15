import sys
import os
STEPS_TO_TRAIN=180000
sys.path.append(os.path.abspath('./'))
from utils import linear_schedule
from custom_callbacks import plot_results
from env_wrappers import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import DQN,SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import EvalCustomCallback
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from env_custom import CartPoleButterHistory, CartPoleButter,CartPoleDiscreteHistory#,CartPoleContinous,CartPoleDiscreteHistory#,CartPoleDiscreteButter2
import argparse

Te=0.02
EP_STEPS=800*0.05/Te
LOAD_MODEL_PATH=None#"./logs/best_model"
LOAD_BUFFER_PATH=None#"dqn_pi_swingup_bufferN"
logdir='./logs/'
VOLTAGE = 12

env = CartPoleButterHistory(Te=Te, N_STEPS=EP_STEPS,discreteActions=True,tensionMax=VOLTAGE, resetMode='experimental',sparseReward=False,Km=0.0,n=5,FILTER=True)#,integrator='ode')#,integrator='rk4')
# env = CartPoleButter(Te=Te,N_STEPS=EP_STEPS,f_a=-7.794018686563599, f_b=0.37538450501353504, f_c=-0.4891760779740128, f_d=-0.002568958116514183,tensionMax=8.47,resetMode='random',sparseReward=False,Km=0.0,n=1)#,integrator='ode')#,integrator='rk4')
env = Monitor(env, filename=logdir+'basic_simulation_')
envEvaluation = env
NORMALISE=False


if NORMALISE:
    ## Automatically normalize the input features and reward
    env1 = DummyVecEnv([lambda: env])
    # env = VecNormalize.load('envNorm.pkl', env)
    env = VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=1000, clip_reward=1000)
    print('using normalised env')
    env.training = True
else:
    envEval=env

#callbacks
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCustomCallback(envEvaluation, best_model_save_path='./logs/',log_path=logdir, eval_freq=15000, n_eval_episodes=30, deterministic=True, render=False)
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)

# from parameters import dqn_sim50
from utils import read_hyperparameters
hyperparams=read_hyperparameters('dqn_cartpole_20')
# from parameters import dqn_sim50
model = DQN(env=env,**hyperparams)
model.gamma = 0.99999
try:
    # model for pendulum starting from bottom
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, callback=[cus_callback, eval_callback])
        if NORMALISE:
            env.training = False
            # reward normalization is not needed at test time
            env.norm_reward = False
            env.save(logdir + 'envNorm.pkl')
        plot_results(logdir)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            env.reset()

finally:
    model.save("deepq_cartpole")
    model.save_replay_buffer('replayBufferDQN')
    # WHEN NORMALISING
    if NORMALISE:
        env.save(logdir + 'envNorm.pkl')
    plot_results(logdir)