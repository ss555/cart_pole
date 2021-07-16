from custom_callbacks import plot_results
import os
import matplotlib.pyplot as plt
x_varArr, y_varArr, legends = plot_results('./weights/dqn',paperMode=True)
plt.plot(x_varArr, y_varArr)
plt.show()