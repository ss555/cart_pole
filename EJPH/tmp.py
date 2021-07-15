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
plt.savefig('zoom.png', dpi = 300, bbox_inches = 'tight', pad_inches = .1)
plt.show()