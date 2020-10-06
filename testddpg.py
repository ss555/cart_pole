import gym

from ddpg import DDPG

env = gym.make('Pendulum-v0')

ddpg = DDPG(
                 env , # Gym environment with continous action space
                 actor(None), # Tensorflow/keras model
                 critic (None), # Tensorflow/keras model
                 buffer (None), # pre-recorded buffer
                 action_bound_range=1,
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 batch_size =64, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 tow = 0.001, # for soft target update
                 discount_factor  = 0.99,
                 explore_time = 1000, # time steps for random actions for exploration
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 dtype = 'float32',
                 n_episodes = 1000 ,# no of episodes to run
                 reward_plot = True ,# (bool)  to plot reward progress per episode
                 model_save = 1) # epochs to save models and buffer

ddpg.train()
