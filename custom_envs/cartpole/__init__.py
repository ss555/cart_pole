import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='cartpoleSwing-v0',
    entry_point='cartpole.envs:CartPoleContinous',
    reward_threshold=10000.0,
    nondeterministic = True,
)

register(
    id='cartpoleSwingD-v0',
    entry_point='cartpole.envs:CartPoleDiscrete',
    reward_threshold=10000.0,
    nondeterministic = True,
)