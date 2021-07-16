import matplotlib.pyplot as plt
import numpy as np

# Create main container
fig = plt.figure()

# Set the random seed
np.random.seed(100)

# Create mock data
x = np.random.normal(400, 50, 10_000)
y = np.random.normal(300, 50, 10_000)
c = np.random.rand(10_000)

def setlabel(ax, label, loc=2, borderpad=0.6, **kwargs):
    legend = ax.get_legend()
    if legend:
        ax.add_artist(legend)
    line, = ax.plot(np.NaN,np.NaN,color='none',label=label)
    label_legend = ax.legend(handles=[line],loc=loc,handlelength=0,handleheight=0,handletextpad=0,borderaxespad=0,borderpad=borderpad,frameon=False,**kwargs)
    label_legend.remove()
    ax.add_artist(label_legend)
    ax.text(-0.2,0.9,chr(98)+')', transform=ax.transAxes)
    line.remove()

# Create zoom-in plot
ax = plt.scatter(x, y, s = 5, c = c)
plt.xlim(400, 500)
plt.ylim(350, 400)
plt.xlabel('x', labelpad = 15)
plt.ylabel('y', labelpad = 15)

# Create zoom-out plot
ax_new = fig.add_axes([0.6, 0.6, 0.2, 0.2]) # the position of zoom-out plot compare to the ratio of zoom-in plot
plt.scatter(x, y, s = 1, c = c)

# Save figure with nice margin
setlabel(ax_new, '(a)')
plt.show()

# for i,ax in enumerate(axs.flat, start=97):
#   ax.plot([0,1],[0,1])
#   ax.text(0.05,0.9,chr(i)+')', transform=ax.transAxes)