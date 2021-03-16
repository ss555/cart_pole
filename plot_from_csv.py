from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils import plot
# obsArr=np.array(pd.read_csv("tmp/back-up-transferLearning/obs-05.csv"))
# timeArr=np.array(pd.read_csv("tmp/back-up-transferLearning/time-05.csv"))
obsArr=np.array(pd.read_csv("tmp/obs.csv"))
timeArr=np.array(pd.read_csv("tmp/time.csv"))
actArr=np.array(pd.read_csv("tmp/actArr.csv"))
# obsArr=np.array(pd.read_csv("tmp/dqn_transfer/obs.csv"))
# timeArr=np.array(pd.read_csv("tmp/dqn_transfer/time.csv"))
# actArr=np.array(pd.read_csv("tmp/dqn_transfer/actArr.csv"))
plot(obsArr,timeArr,actArr,plotlyUse=True)

print('bye')