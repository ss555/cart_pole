import gym
import numpy as np
import yaml
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from env_custom import CartPoleCusBottomDiscrete
env = CartPoleCusBottomDiscrete()
# Use deterministic actions for evaluation and SAVE the best model
eval_callback = EvalCallback(env, best_model_save_path='stable_baselines_dqn_best/',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False)
# Custom MLP policy of two layers of size 32 each with Relu activation function
policy_kwargs = dict(net_arch=[256, 256])
model = DQN(MlpPolicy, env, verbose=0, learning_rate= float(2.3e-3), gradient_steps=128, target_update_interval=10,exploration_fraction=0.16, exploration_final_eps=0.04, buffer_size=100000, learning_starts=10000, batch_size=2048, policy_kwargs=policy_kwargs, tensorboard_log='./dqn_tensorboard/')
'''
#from yml
# Load hyperparameters from yaml file
with open('dqn_config.yml', 'r') as f:
    hyperparams_dict = yaml.safe_load(f)
    hyperparams = hyperparams_dict['CartPoleCusBottom']
    print(hyperparams)
model = DQN(env=env,**hyperparams)
'''
model.learn(total_timesteps=500000, callback=eval_callback)

model.save("dqn_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()