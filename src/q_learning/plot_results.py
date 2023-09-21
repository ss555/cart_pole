#%%
import pandas as pd
pd.options.plotting.backend = "plotly"
import numpy as np
import os
from pathlib import Path
from numba import njit
#%%
path = Path('/home/robotfish/Project/cart_pole/q_learning/0617_102654')
csvfiles = list(path.glob('**/ql_*.csv'))
# print(list(csvfiles))

def plotResult(file):
    os.chdir(file.parent)
    df = pd.read_csv(file)
    df['reward_roll'] = df.reward.rolling(1000).mean()
    df['eps_normalized'] = df['eps']*df.reward.max()
    fig = df.plot(y=['reward_roll', 'eps_normalized'],)
    fig.write_html(file.stem+'_plot.html')

list(map(plotResult, csvfiles))

# %%
path = Path('/home/robotfish/Project/cart_pole/q_learning')
csvfiles = list(path.glob('*.out'))
# print(list(csvfiles))

def plotResult(file):
    os.chdir(file.parent)
    df = pd.read_csv(file, header=None)
    df['reward_roll'] = df[2].rolling(10).mean()
    # df['eps_normalized'] = df[1]*df[2].max()
    fig = df.plot(y=['reward_roll'], title=file.stem)
    fig.show()
    fig.write_html(file.stem+'_plot.html')

#list(map(plotResult, csvfiles))
# %%
