import sys
import os
STEPS_TO_TRAIN=150000
logdir='./logs/other_algo/actions3/'
from sb3_contrib import QRDQN
sys.path.append(os.path.abspath('./'))
from utils import linear_schedule
from custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import ProgressBarManager,SaveOnBestTrainingRewardCallback
# print(os.path.abspath('./'))
from env_custom import CartPoleDiscreteButter#,CartPoleDiscrete5,CartPoleCosSinTensionD,CartPoleCosSinTensionD3,
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
# env = CartPoleCosSinTensionD3(Te=0.05,randomReset=True)
env = CartPoleDiscreteButter(Te=0.05,randomReset=True)#,integrator='rk')
# env = CartPoleDiscrete5(Te=0.05,randomReset=True)
env0 = Monitor(env, logdir)
## Automatically normalize the input features and reward
env1            =DummyVecEnv([lambda:env0])
# env             =VecNormalize(env1, norm_obs=True, norm_reward=True, clip_obs=10000, clip_reward=10000, gamma=0.98)
# envEval         =VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=10000, clip_reward=10000)
# envEval.training=False
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env1, best_model_save_path=logdir,
							 log_path=logdir, eval_freq=5000, n_eval_episodes=20,
							 deterministic=True, render=False)

##my config
model = QRDQN("MlpPolicy", env1, learning_starts=10000, gamma=0.995,
            learning_rate=0.0003993991190316589, exploration_final_eps=0.17787752200089602,
            exploration_fraction=0.08284882874698395, target_update_interval=1000, buffer_size=50000,
            #n_episodes_rollout=1,train_freq=-1,gradient_steps=-1,
              train_freq=256, gradient_steps=256,
              verbose=0, batch_size=2048, policy_kwargs=dict(n_quantiles=10, net_arch=[400, 300]))#net_arch=[128, 128]))

#5actions
# model = QRDQN("MlpPolicy", env1, learning_starts=10000, gamma=0.995,
#             learning_rate=0.0003993991190316589, exploration_final_eps=0.17787752200089602,
#             exploration_fraction=0.08284882874698395, target_update_interval=1000, buffer_size=50000,
#             #n_episodes_rollout=1,train_freq=-1,gradient_steps=-1,
#               train_freq=256, gradient_steps=256,
#               verbose=0, batch_size=2048, policy_kwargs=dict(n_quantiles=10, net_arch=[400, 300]))
NORMALISE=False
# model for pendulum starting from bottom
with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
    model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100,  # tb_log_name="normal",
                callback=[cus_callback, eval_callback, callbackSave])
    if NORMALISE:
        # WHEN NORMALISING
        env.save('envNorm.pkl')
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
    plot_results(logdir)
model.save("deepq_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()
