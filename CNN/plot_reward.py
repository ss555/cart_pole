from custom_callbacks import plot_results
import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./..'))
import matplotlib.pyplot as plt
x_varArr, y_varArr, legends = plot_results('./weights/dqn',paperMode=True)
plt.plot(x_varArr[0], y_varArr[0])
plt.show()