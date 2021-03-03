from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils import plot
obsArr=np.array(pd.read_csv("tmp/back-up-transferLearning/obs-05.csv"))
timeArr=np.array(pd.read_csv("tmp/back-up-transferLearning/time-05.csv"))

plot(obsArr,timeArr)