import sys
import os

STEPS_TO_TRAIN = 300000
logdir = './logs/'
sys.path.append(os.path.abspath('./'))
from utils import linear_schedule
from custom_callbacks import plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn import MlpPolicy
# from stable_baselines3.sac import MlpPolicy
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
# print(os.path.abspath('./'))
from env_custom import CartPoleDiscrete, CartPoleButter  # ,CartPoleContinous,CartPoleDiscreteHistory#,CartPoleDiscreteButter2
import argparse
from stable_baselines3 import HER, DDPG, DQN, SAC, TD3
env = CartPoleButter(Te=0.05, discreteActions=True, resetMode='experimental',
                     sparseReward=True)  # ,integrator='ode')#,integrator='rk4')
# env = CartPoleContinous(Te=0.05,randomReset=True,integrator='ode')#,integrator='rk4')
env0 = Monitor(env, logdir)
## Automatically normalize the input features and reward
# env1 = DummyVecEnv([lambda: env0])
envEvaluation = CartPoleButter(Te=0.05, discreteActions=True, resetMode='experimental', sparseReward=False)
NORMALISE = False

if NORMALISE:
    # env = VecNormalize.load('envNorm.pkl', env)
    env = VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=1000, clip_reward=1000)
    print('using normalised env')
    env.training = True
    # envEval=VecNormalize(env1, norm_obs=True, norm_reward=False, clip_obs=10000, clip_reward=10000)
else:
    envEval = env

# callbacks
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(envEvaluation, best_model_save_path='./logs/',
                             log_path=logdir, eval_freq=15000, n_eval_episodes=30,
                             deterministic=True, render=False)
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
# If True the HER transitions will get sampled online
# online_sampling = True
model_class = DQN  # works also with SAC, DDPG and TD3
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE


# model_class = DQN(MlpPolicy, env, learning_starts=10000, gamma=0.995,
#             learning_rate=0.0003, exploration_final_eps=0.17787752200089602,
#             exploration_initial_eps=0.17787752200089602,
#             target_update_interval=1000, buffer_size=50000, train_freq=256, gradient_steps=256,
#             # n_episodes_rollout=1,train_freq=-1, gradient_steps=-1,
#             verbose=0, batch_size=2024, policy_kwargs=dict(net_arch=[400, 300]))
# model = DQN.load("./logs/best_model", env=env)
# model.exploration_final_eps = 0
# model.exploration_initial_eps = 0
# Initialize the model
online_sampling = True
# Time limit for the episodes
model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy, online_sampling=online_sampling, verbose=1,max_episode_length=env.MAX_STEPS_PER_EPISODE)
try:
    # model for pendulum starting from bottom
    with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
        model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100,  # tb_log_name="normal",
                    callback=[cus_callback, eval_callback])
        if NORMALISE:
            env.training = False
            # reward normalization is not needed at test time
            env.norm_reward = False
            env.save(logdir + 'envNorm.pkl')
        plot_results(logdir)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            env.reset()

finally:
    model.save("deepq_cartpole")
    model.save_replay_buffer('replayBufferDQN')
    # WHEN NORMALISING
    env.save(logdir + 'envNorm.pkl')
    plot_results()