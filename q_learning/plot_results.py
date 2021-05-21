#%%
import pandas as pd
pd.options.plotting.backend = "plotly"
import numpy as np
import os
from pathlib import Path

path = Path('/home/robotfish/Project/cart_pole/q_learning')
csvfiles = path.glob('ql_*.csv')

# %%
