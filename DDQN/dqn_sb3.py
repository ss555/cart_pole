import sys
import os
STEPS_TO_TRAIN=150000
sys.path.append(os.path.abspath('./'))
from utils import linear_schedule
from custom_callbacks import plot_results
from env_wrappers import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import DQN,SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
from env_custom import CartPoleButter,CartPoleDiscreteHistory#,CartPoleContinous,CartPoleDiscreteHistory#,CartPoleDiscreteButter2
import argparse
Te=0.05
EP_STEPS=800
LOAD_MODEL_PATH=None#"./logs/best_model"
LOAD_BUFFER_PATH=None#"dqn_pi_swingup_bufferN"
logdir='./logs/'

env = CartPoleButter(Te=Te,N_STEPS=EP_STEPS,discreteActions=True,tensionMax=8.4706,resetMode='experimental',sparseReward=False,Km=0.0,n=1)#,integrator='ode')#,integrator='rk4')
env = Monitor(env, filename=logdir+'basic_simulation_')
envEvaluation = env
NORMALISE=False

if NORMALISE:
    ## Automatically normalize the input features and reward
    env1 = DummyVecEnv([lambda: env])
    # env = VecNormalize.load('envNorm.pkl', env)
    env=VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=1000, clip_reward=1000)
    print('using normalised env')
    env.training = True
else:
    envEval=env

from pathlib import Path
log_save='./weights/dqn50-sim'
Path(log_save).mkdir(exist_ok=True)
#callbacks
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(envEvaluation, best_model_save_path=log_save,
							 log_path=logdir, eval_freq=15000, n_eval_episodes=30,deterministic=True, render=False)
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
from utils import read_hyperparameters
hyperparams=read_hyperparameters('dqn_cartpole_50')
model = DQN(env=env,**hyperparams)
# model = DQN(MlpPolicy, env, learning_starts=0, gamma=0.995,
#             learning_rate=0.0003, exploration_final_eps=0.17787752200089602, #exploration_initial_eps=0.17787752200089602,
#             target_update_interval=1000, buffer_size=50000, train_freq=(1, "episode"), gradient_steps=-1,#n_episodes_rollout=1,
#             verbose=0, batch_size=1024, policy_kwargs=dict(net_arch=[256, 256]))

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