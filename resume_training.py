import sys
import os
STEPS_TO_TRAIN=300000
logdir='./logs/'
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
from env_custom import CartPoleDiscrete
callbackSave = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)
env = CartPoleDiscrete(Te=0.05,randomReset=True)#,integrator='rk4')
env0 = Monitor(env, logdir)
## Automatically normalize the input features and reward
env1            =DummyVecEnv([lambda:env0])
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env1, best_model_save_path='./logs/training',
							 log_path=logdir, eval_freq=5000, n_eval_episodes=20,
							 deterministic=True, render=False)

model = DQN.load("./logs/best_model", env=env1)
model.learning_rate=0.0003
model.learning_starts=0
model.n_episodes_rollout=1
model.gradient_steps=-1
model.train_freq=-1
model.exploration_final_eps=0.0
model.exploration_initial_eps=model.exploration_final_eps
model.exploration_rate=0.0
# model for pendulum starting from bottom
with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
    model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100,  # tb_log_name="normal",
                callback=[cus_callback, eval_callback])
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

