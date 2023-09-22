import gym
import torch as th
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('/'))

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.env_custom import CartPoleImage

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
# from parameters import dqn_sim50
from src.utils import read_hyperparameters
from stable_baselines3.common.atari_wrappers import AtariWrapper


env = CartPoleImage()
params = read_hyperparameters('atari')
VecFrameStack(params['frame_stack'])
# env1 = AtariWrapper(env)
model = DQN(**params, env=env, tensorboard_log='./CNN/', verbose=1)
# model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, tensorboard_log='./CNN/', verbose=1)
model.learn(10000)
#tensorboard --logdir ./sac_cartpole_tensorboard/