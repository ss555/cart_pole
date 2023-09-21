import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from RLGlue import RLGlue
from src.env_custom import CartPoleRK4
# import plot_script
from cartpoleTileCoder import CartpoleTileCoder
import pandas as pd



test_obs = [[0,0,1,0,0]]

pdtc = CartpoleTileCoder(iht_size=4096, num_tilings=8, num_tiles=4)

result = []
for obs in test_obs:
    tiles = pdtc.get_tiles(obs)
    result.append(tiles)

for tiles in result:
    print(tiles)



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


class QlearningAgent():
    def __init__(self):
        self.tc = None
    def startAgent(self,**agent_info):
        self.numActions = agent_info.get('num_actions')


actor_step_size_arr = [2**(i) for i in range(-6,2,1)]
critic_step_size_arr = [2**(i) for i in range(-4,3,1)]
reward_step_size_arr = [2**(i) for i in range(-11,-1,1)]
num_tilings_arr = [16*i for i in range(1,4,1)]
num_tiles_arr = [8*i for i in range(1,4,1)]

env = CartPoleRK4()#gym.make('Pendulum-v0')
test_agent = QlearningAgent()



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
df = pd.DataFrame(data_list)
df.to_csv('paramsSweep.csv')