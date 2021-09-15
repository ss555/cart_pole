import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset


#  random  walk read

random_walk = np.loadtxt("random_walk_for_pict.txt")


if __name__=='__main__':

    fig = plt.figure(figsize = (8,16))
    ax = plt.subplot(111) #whole path
    ax.plot(random_walk)
    ax.set_xlim(0,5000)
    ax.set_ylim(-130,55)

    axins = zoomed_inset_axes(ax,2,loc='lower right')
    axins.plot(random_walk)

    x1,x2,y1,y2 = 1000,2000, -60,-15
    axins.set_xlim(x1,x2)
    axins.set_ylim(y1,y2)

    mark_inset(ax,axins,loc1=1,loc2=3)
    plt.show()