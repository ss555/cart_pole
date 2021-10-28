import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from RLGlue import RLGlue
from env_custom import CartPoleRK4
# import plot_script
from cartpoleTileCoder import CartpoleTileCoder
import pandas as pd
import os


test_obs = [[0,0,1,0,0]]

pdtc = CartpoleTileCoder(iht_size=4096, num_tilings=8, num_tiles=4)

result = []
for obs in test_obs:
    tiles = pdtc.get_tiles(obs)
    result.append(tiles)

for tiles in result:
    print(tiles)


def compute_softmax_prob(actor_w, tiles):
    """
    Computes softmax probability for all actions

    Args:
    actor_w - np.array, an array of actor weights
    tiles - np.array, an array of active tiles

    Returns:
    softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
    """

    # First compute the list of state-action preferences (1~2 lines)
    # state_action_preferences = ? (list of size 3)
    ### START CODE HERE ###
    state_action_preferences = actor_w[:, tiles].sum(axis=1)
    ### END CODE HERE ###

    # Set the constant c by finding the maximum of state-action preferences (use np.max) (1 line)
    # c = ? (float)
    ### START CODE HERE ###
    c = np.max(state_action_preferences)
    ### END CODE HERE ###

    # Compute the numerator by subtracting c from state-action preferences and exponentiating it (use np.exp) (1 line)
    # numerator = ? (list of size 3)
    ### START CODE HERE ###
    numerator = np.exp(state_action_preferences - c)
    ### END CODE HERE ###

    # Next compute the denominator by summing the values in the numerator (use np.sum) (1 line)
    # denominator = ? (float)
    ### START CODE HERE ###
    denominator = np.sum(numerator)
    ### END CODE HERE ###

    # Create a probability array by dividing each element in numerator array by denominator (1 line)
    # We will store this probability array in self.softmax_prob as it will be useful later when updating the Actor
    # softmax_prob = ? (list of size 3)
    ### START CODE HERE ###
    softmax_prob = numerator / denominator
    ### END CODE HERE ###

    return softmax_prob

# set tile-coder
iht_size = 4096
num_tilings = 8
num_tiles = 8
test_tc = CartpoleTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

num_actions = 3
actions = list(range(num_actions))
actor_w = np.zeros((len(actions), iht_size))

# setting actor weights such that state-action preferences are always [-1, 1, 2]
actor_w[0] = -1./num_tilings
actor_w[1] = 1./num_tilings
actor_w[2] = 2./num_tilings

# obtain active_tiles from state
state = [0,0,1,0,0]
active_tiles = test_tc.get_tiles(state)

# compute softmax probability
softmax_prob = compute_softmax_prob(actor_w, active_tiles)
print('softmax probability: {}'.format(softmax_prob))


class ActorCriticSoftmaxAgent():
    def __init__(self):
        self.rand_generator = None

        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None

        self.tc = None

        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None

        self.actions = None

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            "iht_size": int
            "num_tilings": int,
            "num_tiles": int,
            "actor_step_size": float,
            "critic_step_size": float,
            "avg_reward_step_size": float,
            "num_actions": int,
            "seed": int
        }
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        iht_size = agent_info.get("iht_size")
        num_tilings = agent_info.get("num_tilings")
        num_tiles = agent_info.get("num_tiles")

        # initialize self.tc to the tile coder we created
        self.tc = CartpoleTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        # set step-size accordingly (we normally divide actor and critic step-size by num. tilings (p.217-218 of textbook))
        self.actor_step_size = agent_info.get("actor_step_size") / num_tilings
        self.critic_step_size = agent_info.get("critic_step_size") / num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")

        self.actions = list(range(agent_info.get("num_actions")))

        # Set initial values of average reward, actor weights, and critic weights
        # We initialize actor weights to three times the iht_size.
        # Recall this is because we need to have one set of weights for each of the three actions.
        self.avg_reward = 0.0
        self.actor_w = np.zeros((len(self.actions), iht_size))
        self.critic_w = np.zeros(iht_size)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_policy(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder

        Returns:
            The action selected according to the policy
        """

        # compute softmax probability
        softmax_prob = compute_softmax_prob(self.actor_w, active_tiles)

        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)

        # save softmax_prob as it will be useful later when updating the Actor
        self.softmax_prob = softmax_prob

        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        active_tiles = self.tc.get_tiles(state)
        current_action = self.agent_policy(active_tiles)
        ### END CODE HERE ###

        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """

        ### Use self.tc to get active_tiles using angle and ang_vel (1 line)
        ### START CODE HERE ###
        active_tiles = self.tc.get_tiles(state)
        ### END CODE HERE ###

        ### Compute delta using Equation (1) (1 line)
        # delta = ?
        ### START CODE HERE ###
        delta = reward - self.avg_reward + self.critic_w[active_tiles].sum() - self.critic_w[self.prev_tiles].sum()
        ### END CODE HERE ###

        ### update average reward using Equation (2) (1 line)
        # self.avg_reward += ?
        ### START CODE HERE ###
        self.avg_reward += self.avg_reward_step_size * delta
        ### END CODE HERE ###

        # update critic weights using Equation (3) and (5) (1 line)
        # self.critic_w[self.prev_tiles] += ?
        ### START CODE HERE ###
        self.critic_w[self.prev_tiles] += self.critic_step_size * delta
        ### END CODE HERE ###

        # update actor weights using Equation (4) and (6)
        # We use self.softmax_prob saved from the previous timestep
        # We leave it as an exercise to verify that the code below corresponds to the equation.
        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (0 - self.softmax_prob[a])

        ### set current_action by calling self.agent_policy with active_tiles (1 line)
        # current_action = ?
        ### START CODE HERE ###
        current_action = self.agent_policy(active_tiles)
        ### END CODE HERE ###

        self.prev_tiles = active_tiles
        self.last_action = current_action

        return self.last_action

    def agent_message(self, message):
        if message == 'get avg reward':
            return self.avg_reward



actor_step_size_arr = [2**(i) for i in range(-6,2,1)]
critic_step_size_arr = [2**(i) for i in range(-4,3,1)]
reward_step_size_arr = [2**(i) for i in range(-11,-1,1)]
num_tilings_arr = [16*i for i in range(1,4,1)]
num_tiles_arr = [8*i for i in range(1,4,1)]

env = CartPoleRK4()#gym.make('Pendulum-v0')
test_agent = ActorCriticSoftmaxAgent()



#train
def train(env,test_agent, plotReward=False):
    NUM_STEPS_TRAIN = int(1e5)
    action = test_agent.agent_start(env.reset())
    rewArr=[]
    epArr=0
    for _ in range(NUM_STEPS_TRAIN):
        obs,rew,done,_ = env.step(action)
        action = test_agent.agent_step(rew,obs)
        epArr+=rew
        if done:
            rewArr.append(epArr)
            epArr=0
            action = test_agent.agent_start(env.reset())
    if plotReward:
        plt.plot(rewArr)
        plt.show()
    print(f'end{np.mean(rewArr[-10:])}')
    return rewArr
data_list=[]
try:
    for num_tiles in num_tiles_arr:
        for num_tilings in num_tilings_arr:
            for ac_step in actor_step_size_arr:
                for cr_step in critic_step_size_arr:
                    for re_step in reward_step_size_arr:
                        agent_info = {
                            "num_tilings": num_tiles,
                            "num_tiles": num_tilings,
                            "actor_step_size": ac_step,
                            "critic_step_size": cr_step,
                            "avg_reward_step_size": re_step,
                            "num_actions": 3,
                            "iht_size": 4096 * 128
                        }
                        test_agent.agent_init(agent_info)
                        try:
                            reArr = train(env,test_agent)
                        except:
                            reArr = [np.NaN]

                        data_list.append({'num_tilings': num_tiles, 'num_tiles': num_tilings,
                                          'actor_step_size': ac_step, 'critic_step_size': cr_step,
                                 'avg_reward_step_size': re_step, 'reward': reArr})
except Exception as e:
    print(e)
finally:
    df = pd.DataFrame(data_list)
    log_path = './q_learning/df/'

    os.makedirs(log_path,exist_ok=True)
    df.to_csv(log_path+'paramsSweep.csv')