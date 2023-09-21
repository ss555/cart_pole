from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils import plot

path='/media/sardor/b1/12-STABLE3/weights/sac50-real/pwm255.npz'
data=np.load(path)
print(data.files)
fig, ax = plt.subplots(1, 1)
ax.plot(data[data.files[0]], data[data[data.files[1]]], 'r.')
plt.show()

print('bye')