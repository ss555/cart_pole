import os
import gym
import pybullet_envs
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3 import PPO
from custom_callbacks import ProgressBarManager
from stable_baselines3.common.callbacks import EvalCallback
import pybullet as p


STEPS_TO_TRAIN=1000
# env=gym.make("InvertedPendulumBulletEnv-v0")
env = DummyVecEnv([lambda: gym.make("InvertedPendulumBulletEnv-v0")]) # Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
model = PPO('MlpPolicy', env)
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env, best_model_save_path='../logs/',
                             log_path='../logs/', eval_freq=100,
                             deterministic=True, render=False)
# model for pendulum starting from bottom
with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
    model.learn(total_timesteps=STEPS_TO_TRAIN, log_interval=100, tb_log_name="500K_bottom",callback=[cus_callback, eval_callback])
# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "tmp/"
model.save(log_dir + "ppo_cart")
stats_path = os.path.join(log_dir, "vec_normalize_cart.pkl")
env.save(stats_path)
# To demonstrate loading del model, env
# Load the agent model = PPO2.load(log_dir + "ppo_halfcheetah")
# Load the saved statistics

# env = DummyVecEnv([lambda: gym.make("InvertedPendulumBulletEnv-v0")])
# env = VecNormalize.load(stats_path, env) # do not update them at test time
# env.training = False # reward normalization is not needed at test time
# env.norm_reward = False

env.render(mode="human")

#visualise
p.connect(p.GUI)
p.setGravity(0, 0, -10)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()